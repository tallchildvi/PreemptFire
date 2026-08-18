import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
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
from src.data_pipeline.accessibility.solver import dijkstra_kernel


class SpatialFeatureFetcher:
    """
    fetches terrain dem, calculates native gradients, queries buffered osm infrastructure,
    and computes anisotropic hiking accessibility and euclidean distance layers
    """

    def __init__(self):
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        self.aligner = GridAligner()

        # osm tag catalogs
        self.tags_roads = {
            "highway": [
                "motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service",
                "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"
            ]
        }
        self.tags_trails = {
            "highway": ["path", "footway", "track", "bridleway", "cycleway", "pedestrian"]
        }
        self.tags_railways = {
            "railway": ["rail", "narrow_gauge", "spur", "yard"]
        }
        self.tags_camps = {
            "tourism": ["camp_site", "picnic_site", "caravan_site", "wilderness_hut", "alpine_hut", "chalet", "viewpoint"],
            "amenity": ["shelter", "bbq", "firepit"],
            "leisure": ["firepit", "picnic_table"]
        }
        self.tags_power = {
            "power": ["line", "minor_line", "cable", "substation", "transformer"]
        }

    def _fetch_mosaicked_dem_30m_utm(self, grid_info: dict) -> tuple[np.ndarray, rasterio.Affine]:
        """fetches all intersecting copernicus 30m dem tiles and reprojects to native 30m utm"""
        bbox = grid_info["bbox_wgs84"]
        search = self.stac_client.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox
        )
        items = list(search.items())
        if not items:
            raise RuntimeError(f"no copernicus dem items found for bbox: {bbox}")

        src_files = [rasterio.open(item.assets["data"].href) for item in items]
        try:
            mosaic_arr, mosaic_transform = merge(src_files)
            src_crs = src_files[0].crs
        finally:
            for f in src_files:
                f.close()

        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        native_size = int(round(20480.0 / 30.0))  # ~683 px at 30m
        native_transform = from_bounds(min_x, min_y, max_x, max_y, native_size, native_size)

        dem_30m_utm = np.full((native_size, native_size), np.nan, dtype=np.float32)

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
        """computes physical slope and compass aspect on native 30m grid, resamples to 10m master grid"""
        dem_30m, native_transform = self._fetch_mosaicked_dem_30m_utm(grid_info)
        pixel_size_x = abs(native_transform.a)
        pixel_size_y = abs(native_transform.e)

        # row index increases southward, so dz_d_north = -dz_d_south
        dz_d_south, dz_d_east = np.gradient(dem_30m, pixel_size_y, pixel_size_x)
        dz_d_north = -dz_d_south

        slope_rad = np.arctan(np.sqrt(dz_d_east**2 + dz_d_north**2))
        slope_deg_30m = np.degrees(slope_rad).astype(np.float32)

        downslope_east = -dz_d_east
        downslope_north = -dz_d_north
        aspect_rad = np.arctan2(downslope_east, downslope_north) % (2.0 * np.pi)

        northness_30m = np.cos(aspect_rad).astype(np.float32)
        eastness_30m = np.sin(aspect_rad).astype(np.float32)

        flat_mask = slope_deg_30m < 0.1
        northness_30m[flat_mask] = 0.0
        eastness_30m[flat_mask] = 0.0

        terrain_layers_30m = {
            "Elevation": dem_30m,
            "Slope": slope_deg_30m,
            "Northness": northness_30m,
            "Eastness": eastness_30m
        }

        aligned_terrain = {}
        for name, arr_30m in terrain_layers_30m.items():
            aligned_terrain[name] = self.aligner.align_raster_to_master(
                src_array=arr_30m,
                src_crs=grid_info["crs"],
                src_transform=native_transform,
                grid_info=grid_info,
                resampling_method=Resampling.bilinear,
                dst_nodata=np.nan
            )

        return aligned_terrain

    def _query_osm_buffered(self, grid_info: dict, osm_tags: dict, buffer_meters: float = 5000.0):
        """queries osm vectors within a 5km utm buffer and projects to local utm"""
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        utm_crs = grid_info["crs"]

        b_min_x, b_max_x = min_x - buffer_meters, max_x + buffer_meters
        b_min_y, b_max_y = min_y - buffer_meters, max_y + buffer_meters

        transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        corners = [
            (b_min_x, b_min_y), (b_min_x, b_max_y),
            (b_max_x, b_min_y), (b_max_x, b_max_y)
        ]
        corners_wgs84 = [transformer.transform(x, y) for x, y in corners]
        b_lons = [pt[0] for pt in corners_wgs84]
        b_lats = [pt[1] for pt in corners_wgs84]
        buffered_wgs84_bbox = (min(b_lons), min(b_lats), max(b_lons), max(b_lats))

        west, south, east, north = buffered_wgs84_bbox
        try:
            if hasattr(ox, "features_from_bbox"):
                gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=osm_tags)
            else:
                gdf = ox.geometries_from_bbox(north, south, east, west, tags=osm_tags)

            if gdf is None or gdf.empty:
                return None
            return gdf.to_crs(utm_crs)
        except Exception as e:
            print(f"osm query warning for {list(osm_tags.keys())}: {e}")
            return None

    def _compute_buffered_edt(
        self,
        gdf_utm,
        grid_info: dict,
        buffer_meters: float = 5000.0
    ) -> np.ndarray:
        """computes euclidean distance in meters with buffer and crops to master grid"""
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]

        b_min_x, b_max_x = min_x - buffer_meters, max_x + buffer_meters
        b_min_y, b_max_y = min_y - buffer_meters, max_y + buffer_meters

        buf_px_x = int(round((b_max_x - b_min_x) / 10.0))
        buf_px_y = int(round((b_max_y - b_min_y) / 10.0))
        buf_transform = from_bounds(b_min_x, b_min_y, b_max_x, b_max_y, buf_px_x, buf_px_y)

        if gdf_utm is None or gdf_utm.empty:
            # no infrastructure in wilderness -> return large default distance
            return np.full(grid_info["shape"], 50000.0, dtype=np.float32)

        shapes = [(geom, 1) for geom in gdf_utm.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            return np.full(grid_info["shape"], 50000.0, dtype=np.float32)

        buffered_mask = rasterize(
            shapes=shapes,
            out_shape=(buf_px_y, buf_px_x),
            transform=buf_transform,
            fill=0,
            dtype=np.uint8
        )

        dist_meters_buf = distance_transform_edt(1 - buffered_mask) * 10.0

        offset_x = int(round(buffer_meters / 10.0))
        offset_y = int(round(buffer_meters / 10.0))

        master_dist = dist_meters_buf[
            offset_y : offset_y + GRID_SIZE_PX,
            offset_x : offset_x + GRID_SIZE_PX
        ].astype(np.float32)

        return master_dist


    def _compute_accessibility_channel(
        self,
        elevation_10m: np.ndarray,
        passable_mask: np.ndarray,
        gdf_utm,
        grid_info: dict,
        resolution: float = 10.0,
        max_time_cap_hours: float = 12.0
    ) -> np.ndarray:
        """computes physical walking time (hours) via tobler dijkstra with water barrier constraints"""
        shape = grid_info["shape"]
        transform = grid_info["transform"]

        sources = np.zeros(shape, dtype=np.uint8)
        if gdf_utm is not None and not gdf_utm.empty:
            shapes = [(geom, 1) for geom in gdf_utm.geometry if geom is not None and not geom.is_empty]
            if shapes:
                sources = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)

        if not np.any((sources == 1) & (passable_mask == 1)):
            return np.full(shape, max_time_cap_hours, dtype=np.float32)

        # run dijkstra kernel
        optimal_time = dijkstra_kernel(elevation_10m, sources, passable_mask, resolution)

        # cap inf and unreachable areas at maximum time cost
        clean_time = np.where(np.isinf(optimal_time), max_time_cap_hours, optimal_time)
        clean_time = np.clip(clean_time, 0.0, max_time_cap_hours).astype(np.float32)

        return clean_time


    def fetch_all_spatial_features(
        self,
        lat: float,
        lon: float,
        scl_10m: np.ndarray = None
    ) -> dict:
        """
        master pipeline method: collects all 9 spatial context layers:
        - 4 terrain metrics: Elevation, Slope, Northness, Eastness
        - 2 accessibility times: Travel_Time_Roads, Travel_Time_Trails
        - 3 euclidean distances: Dist_to_Railways, Dist_to_Camps, Dist_to_Powerlines
        """
        grid_info = self.aligner.get_master_grid_info(lat, lon)
        shape = grid_info["shape"]
        transform = grid_info["transform"]

        # Step 1: fetch elevation and compute slope/aspect
        terrain = self.fetch_dem_features(grid_info)
        elevation_10m = terrain["Elevation"]

        # Step 2: query buffered osm layers
        gdf_roads = self._query_osm_buffered(grid_info, self.tags_roads)
        gdf_trails = self._query_osm_buffered(grid_info, self.tags_trails)
        gdf_railways = self._query_osm_buffered(grid_info, self.tags_railways)
        gdf_camps = self._query_osm_buffered(grid_info, self.tags_camps)
        gdf_power = self._query_osm_buffered(grid_info, self.tags_power)

        # Step 3: build passable mask with bridge punch-through
        passable_mask = np.ones(shape, dtype=np.uint8)
        if scl_10m is not None:
            passable_mask[scl_10m == 6] = 0  # SCL class 6 = water
        passable_mask[np.isnan(elevation_10m)] = 0

        # roads and trails unblock water pixels for valid bridges
        bridge_shapes = []
        for gdf in [gdf_roads, gdf_trails]:
            if gdf is not None and not gdf.empty:
                bridge_shapes.extend([(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty])
        if bridge_shapes:
            bridges_mask = rasterize(bridge_shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)
            passable_mask[bridges_mask == 1] = 1

        # Step 4: compute accessibility surfaces (tobler hiking hours)
        travel_roads = self._compute_accessibility_channel(elevation_10m, passable_mask, gdf_roads, grid_info)
        travel_trails = self._compute_accessibility_channel(elevation_10m, passable_mask, gdf_trails, grid_info)

        # Step 5: compute euclidean distance maps (meters)
        dist_railways = self._compute_buffered_edt(gdf_railways, grid_info)
        dist_camps = self._compute_buffered_edt(gdf_camps, grid_info)
        dist_power = self._compute_buffered_edt(gdf_power, grid_info)

        return {
            "Elevation": elevation_10m,
            "Slope": terrain["Slope"],
            "Northness": terrain["Northness"],
            "Eastness": terrain["Eastness"],
            "Travel_Time_Roads": travel_roads,
            "Travel_Time_Trails": travel_trails,
            "Dist_to_Railways": dist_railways,
            "Dist_to_Camps": dist_camps,
            "Dist_to_Powerlines": dist_power
        }


if __name__ == "__main__":
    fetcher = SpatialFeatureFetcher()
    # test fetch in alberta
    data = fetcher.fetch_all_spatial_features(lat=53.5461, lon=-113.4937)

    for name, arr in data.items():
        print(f"{name:20s} | shape: {arr.shape} | range: [{np.nanmin(arr):8.2f}, {np.nanmax(arr):8.2f}]")