import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer

import osmnx as ox

from src.config import GRID_SIZE_PX
from src.processing.grid_aligner import GridAligner


class OsmDemFetcher:
    """fetches copernicus dem, computes native-resolution terrain metrics, and builds buffered osm distance maps"""

    def __init__(self):
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        self.aligner = GridAligner()

    def _fetch_mosaicked_dem_30m_utm(self, grid_info: dict) -> tuple[np.ndarray, rasterio.Affine]:
        """
        fetches all intersecting copernicus 30m dem tiles, merges them, 
        and reprojects to local utm at native 30m resolution
        """
        bbox = grid_info["bbox_wgs84"]
        search = self.stac_client.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox
        )
        items = list(search.items())
        if not items:
            raise RuntimeError(f"no copernicus dem items found for bbox: {bbox}")

        # open all intersecting dem tile streams
        src_files = [rasterio.open(item.assets["data"].href) for item in items]
        try:
            # merge tiles into one continuous wgs84 array
            mosaic_arr, mosaic_transform = merge(src_files)
            src_crs = src_files[0].crs
        finally:
            for f in src_files:
                f.close()

        # define native 30m utm grid over the exact 20.48 km extent
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        native_size = int(round(20480.0 / 30.0))  # ~683 px at 30m
        native_transform = from_bounds(min_x, min_y, max_x, max_y, native_size, native_size)

        dem_30m_utm = np.full((native_size, native_size), np.nan, dtype=np.float32)

        # reproject wgs84 mosaic into native 30m utm
        reproject(
            source=mosaic_arr[0],
            destination=dem_30m_utm,
            src_transform=mosaic_transform,
            src_crs=src_crs,
            dst_transform=native_transform,
            dst_crs=grid_info["crs"],
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )

        return dem_30m_utm, native_transform

    def fetch_dem_features(self, grid_info: dict) -> dict:
        """
        computes elevation, slope, northness, and eastness at native 30m resolution,
        then resamples all continuous terrain metrics to the 10m master grid
        """
        dem_30m, native_transform = self._fetch_mosaicked_dem_30m_utm(grid_info)
        pixel_size_x = abs(native_transform.a)
        pixel_size_y = abs(native_transform.e)

        # 1. compute gradients in physical meters on native 30m grid
        # axis 0 goes south (increasing row index), axis 1 goes east
        dz_d_south, dz_d_east = np.gradient(dem_30m, pixel_size_y, pixel_size_x)
        dz_d_north = -dz_d_south

        # 2. physical slope in degrees
        slope_rad = np.arctan(np.sqrt(dz_d_east**2 + dz_d_north**2))
        slope_deg_30m = np.degrees(slope_rad).astype(np.float32)

        # 3. physical compass aspect (direction of steepest descent)
        downslope_east = -dz_d_east
        downslope_north = -dz_d_north
        aspect_rad = np.arctan2(downslope_east, downslope_north) % (2.0 * np.pi)

        # 4. continuous northness (+1 north, -1 south) and eastness (+1 east, -1 west)
        northness_30m = np.cos(aspect_rad).astype(np.float32)
        eastness_30m = np.sin(aspect_rad).astype(np.float32)

        # flat terrain (zero slope) defaults to neutral 0.0 exposure
        flat_mask = slope_deg_30m < 0.1
        northness_30m[flat_mask] = 0.0
        eastness_30m[flat_mask] = 0.0

        # 5. resample all 4 layers from native 30m utm to 10m 2048x2048 master grid
        terrain_layers = {
            "Elevation": dem_30m,
            "Slope": slope_deg_30m,
            "Northness": northness_30m,
            "Eastness": eastness_30m
        }

        aligned_terrain = {}
        for name, arr_30m in terrain_layers.items():
            aligned_terrain[name] = self.aligner.align_raster_to_master(
                src_array=arr_30m,
                src_crs=grid_info["crs"],
                src_transform=native_transform,
                grid_info=grid_info,
                resampling_method=Resampling.bilinear,
                dst_nodata=np.nan
            )

        return aligned_terrain

    def _query_osm_features(self, buffered_wgs84_bbox: list, tags: dict):
        """robust osm query compatible with osmnx 1.x and 2.x"""
        west, south, east, north = buffered_wgs84_bbox
        try:
            # osmnx 2.x api
            if hasattr(ox, "features_from_bbox"):
                return ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
            # osmnx 1.x fallback
            return ox.geometries_from_bbox(north, south, east, west, tags=tags)
        except Exception as e:
            print(f"osm query warning for tags {tags}: {e}")
            return None

    def fetch_osm_decay_distance(
        self, 
        grid_info: dict, 
        osm_tags: dict, 
        buffer_meters: float = 5000.0
    ) -> np.ndarray:
        """
        queries osm with a 5km spatial buffer, computes euclidean distance transform
        on the expanded grid, and crops back to the exact 10m master grid
        """
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        utm_crs = grid_info["crs"]

        # 1. expand bounds by buffer (5 km) in utm meters
        b_min_x = min_x - buffer_meters
        b_max_x = max_x + buffer_meters
        b_min_y = min_y - buffer_meters
        b_max_y = max_y + buffer_meters

        buf_px_x = int(round((b_max_x - b_min_x) / 10.0))  # 2048 + 1000 = 3048 px
        buf_px_y = int(round((b_max_y - b_min_y) / 10.0))

        buf_transform = from_bounds(b_min_x, b_min_y, b_max_x, b_max_y, buf_px_x, buf_px_y)

        # 2. convert buffered utm bounds to wgs84 for overpass query
        transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        corners = [
            (b_min_x, b_min_y), (b_min_x, b_max_y),
            (b_max_x, b_min_y), (b_max_x, b_max_y)
        ]
        corners_wgs84 = [transformer.transform(x, y) for x, y in corners]
        b_lons = [pt[0] for pt in corners_wgs84]
        b_lats = [pt[1] for pt in corners_wgs84]
        buffered_wgs84_bbox = [min(b_lons), min(b_lats), max(b_lons), max(b_lats)]

        # 3. query osm
        gdf = self._query_osm_features(buffered_wgs84_bbox, osm_tags)
        
        # if network failed or error occurred, do NOT assume zero distance
        if gdf is None:
            return np.full(grid_info["shape"], np.nan, dtype=np.float32)

        # if query succeeded but no infrastructure exists in deep wilderness, dist is max
        if gdf.empty:
            return np.zeros(grid_info["shape"], dtype=np.float32)

        # 4. project to utm and rasterize on buffered grid
        gdf_utm = gdf.to_crs(utm_crs)
        shapes = [(geom, 1) for geom in gdf_utm.geometry if geom is not None and not geom.is_empty]
        
        if not shapes:
            return np.zeros(grid_info["shape"], dtype=np.float32)

        buffered_mask = rasterize(
            shapes=shapes,
            out_shape=(buf_px_y, buf_px_x),
            transform=buf_transform,
            fill=0,
            dtype=np.uint8
        )

        # 5. euclidean distance transform (pixel distance * 10m)
        dist_meters_buf = distance_transform_edt(1 - buffered_mask) * 10.0
        decay_buf = np.exp(-dist_meters_buf / 1000.0).astype(np.float32)

        # 6. crop central 2048x2048 region matching exact master grid
        crop_offset_x = int(round(buffer_meters / 10.0))  # 500 px
        crop_offset_y = int(round(buffer_meters / 10.0))

        master_decay = decay_buf[
            crop_offset_y : crop_offset_y + GRID_SIZE_PX,
            crop_offset_x : crop_offset_x + GRID_SIZE_PX
        ]

        return master_decay

    def fetch_all_spatial_context(self, lat: float, lon: float) -> dict:
        """collects terrain features and anthropogenic decay maps for a master scene"""
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        # 1. topography features (native 30m -> 10m master grid)
        terrain = self.fetch_dem_features(grid_info)

        # 2. anthropogenic osm decay layers (5km buffered query -> 10m cropped master grid)
        dist_roads = self.fetch_osm_decay_distance(
            grid_info, {"highway": ["primary", "secondary", "tertiary", "trunk", "motorway", "residential"]}
        )
        dist_trails = self.fetch_osm_decay_distance(
            grid_info, {"highway": ["path", "footway", "track"]}
        )
        dist_camps = self.fetch_osm_decay_distance(
            grid_info, {"tourism": ["camp_site", "picnic_site"]}
        )
        dist_power = self.fetch_osm_decay_distance(
            grid_info, {"power": ["line", "minor_line"]}
        )

        return {
            **terrain,
            "Dist_to_Roads": dist_roads,
            "Dist_to_Trails": dist_trails,
            "Dist_to_Camps": dist_camps,
            "Dist_to_Powerlines": dist_power
        }


if __name__ == "__main__":
    fetcher = OsmDemFetcher()
    # test fetch for a complex terrain scene in alberta
    data = fetcher.fetch_all_spatial_context(lat=53.5461, lon=-113.4937)

    print(f"Elevation shape: {data['Elevation'].shape}, Range: [{np.nanmin(data['Elevation']):.1f}, {np.nanmax(data['Elevation']):.1f}] m")
    print(f"Slope shape:     {data['Slope'].shape}, Range: [{np.nanmin(data['Slope']):.1f}, {np.nanmax(data['Slope']):.1f}] deg")
    print(f"Northness range: [{np.nanmin(data['Northness']):.2f}, {np.nanmax(data['Northness']):.2f}]")
    print(f"Dist_to_Roads:   [{np.nanmin(data['Dist_to_Roads']):.3f}, {np.nanmax(data['Dist_to_Roads']):.3f}]")