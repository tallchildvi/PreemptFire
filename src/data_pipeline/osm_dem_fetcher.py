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
    """fetches terrain, osm infrastructure, accessibility and distance features."""

    def __init__(self):
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        self.aligner = GridAligner()

        self.tags_water = {
            "natural": ["water", "wetland"],
            "waterway": ["river", "riverbank", "stream", "canal"]
        }

        self.tags_roads = {
            "highway": [
                "motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service",
                "motorway_link", "trunk_link", "primary_link",
                "secondary_link", "tertiary_link"
            ]
        }

        self.tags_trails = {
            "highway": [
                "path", "footway", "track",
                "bridleway", "cycleway", "pedestrian"
            ]
        }

        self.tags_railways = {
            "railway": ["rail", "narrow_gauge", "spur", "yard"]
        }

        self.tags_camps = {
            "tourism": [
                "camp_site", "picnic_site", "caravan_site",
                "wilderness_hut", "alpine_hut", "chalet", "viewpoint"
            ],
            "amenity": ["shelter", "bbq", "firepit"],
            "leisure": ["firepit", "picnic_table"]
        }

        self.tags_power = {
            "power": [
                "line", "minor_line", "cable",
                "substation", "transformer"
            ]
        }

        self.tags_bridges = {
            "bridge": ["yes"]
        }

    def _get_buffered_grid(self, grid_info: dict, buffer_meters: float, resolution: float) -> tuple:
        """returns buffered utm bounds, shape and transform."""

        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]

        b_min_x = min_x - buffer_meters
        b_max_x = max_x + buffer_meters
        b_min_y = min_y - buffer_meters
        b_max_y = max_y + buffer_meters

        width = int(round((b_max_x - b_min_x) / resolution))
        height = int(round((b_max_y - b_min_y) / resolution))

        transform = from_bounds(b_min_x, b_min_y, b_max_x, b_max_y, width, height)

        return (
            (b_min_x, b_min_y, b_max_x, b_max_y),
            (height, width),
            transform
        )

    def _get_buffered_wgs84_bbox(self, grid_info: dict, buffer_meters: float) -> tuple:
        """returns buffered wgs84 bbox from the utm scene geometry."""

        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]

        b_min_x = min_x - buffer_meters
        b_max_x = max_x + buffer_meters
        b_min_y = min_y - buffer_meters
        b_max_y = max_y + buffer_meters

        transformer = Transformer.from_crs(grid_info["crs"], "EPSG:4326", always_xy=True)

        corners = [
            (b_min_x, b_min_y),
            (b_min_x, b_max_y),
            (b_max_x, b_min_y),
            (b_max_x, b_max_y)
        ]

        corners_wgs84 = [
            transformer.transform(x, y)
            for x, y in corners
        ]

        lons = [point[0] for point in corners_wgs84]
        lats = [point[1] for point in corners_wgs84]

        return (
            min(lons),
            min(lats),
            max(lons),
            max(lats)
        )

    def _fetch_dem_raster(self, grid_info: dict, buffer_meters: float, resolution: float) -> tuple[np.ndarray, rasterio.Affine]:
        """fetches copernicus dem and reprojects it to the target utm grid."""

        if buffer_meters == 0:
            bbox = grid_info["bbox_wgs84"]
        else:
            bbox = self._get_buffered_wgs84_bbox(grid_info, buffer_meters)

        search = self.stac_client.search(collections=["cop-dem-glo-30"], bbox=bbox)

        items = list(search.items())

        if not items:
            raise RuntimeError(f"no copernicus dem items found for bbox: {bbox}")

        src_files = [
            rasterio.open(item.assets["data"].href)
            for item in items
        ]

        try:
            mosaic_arr, mosaic_transform = merge(src_files)
            src_crs = src_files[0].crs
            src_nodata = src_files[0].nodata
        finally:
            for src in src_files:
                src.close()

        _, target_shape, target_transform = self._get_buffered_grid(grid_info, buffer_meters, resolution)

        dem = np.full(target_shape, np.nan, dtype=np.float32)

        reproject_kwargs = {
            "source": mosaic_arr[0],
            "destination": dem,
            "src_transform": mosaic_transform,
            "src_crs": src_crs,
            "dst_transform": target_transform,
            "dst_crs": grid_info["crs"],
            "resampling": Resampling.bilinear,
            "dst_nodata": np.nan,
        }

        if src_nodata is not None:
            reproject_kwargs["src_nodata"] = src_nodata

        reproject(**reproject_kwargs)

        return dem, target_transform

    def fetch_dem_features(self, grid_info: dict) -> dict:
        """computes terrain features on native-scale dem and aligns them."""

        dem_30m, native_transform = self._fetch_dem_raster(grid_info, buffer_meters=0.0, resolution=30.0)

        pixel_size_x = abs(native_transform.a)
        pixel_size_y = abs(native_transform.e)

        dz_d_south, dz_d_east = np.gradient(dem_30m, pixel_size_y, pixel_size_x)

        dz_d_north = -dz_d_south

        slope_rad = np.arctan(
            np.sqrt(
                dz_d_east ** 2 +
                dz_d_north ** 2
            )
        )

        slope_deg = np.degrees(slope_rad).astype(np.float32)

        downslope_east = -dz_d_east
        downslope_north = -dz_d_north

        aspect_rad = (
            np.arctan2(
                downslope_east,
                downslope_north
            ) % (2.0 * np.pi)
        )

        northness = np.cos(aspect_rad).astype(np.float32)
        eastness = np.sin(aspect_rad).astype(np.float32)

        flat_mask = slope_deg < 0.1
        northness[flat_mask] = 0.0
        eastness[flat_mask] = 0.0

        terrain_layers = {
            "Elevation": dem_30m,
            "Slope": slope_deg,
            "Northness": northness,
            "Eastness": eastness
        }

        aligned = {}

        for name, arr in terrain_layers.items():
            aligned[name] = self.aligner.align_raster_to_master(
                src_array=arr,
                src_crs=grid_info["crs"],
                src_transform=native_transform,
                grid_info=grid_info,
                resampling_method=Resampling.bilinear,
                dst_nodata=np.nan
            )

        return aligned

    def _query_osm_buffered(self, grid_info: dict, osm_tags: dict, buffer_meters: float = 5000.0):
        """queries osm within a metric buffer and projects geometries to utm."""

        bbox = self._get_buffered_wgs84_bbox(grid_info, buffer_meters)

        west, south, east, north = bbox

        try:
            if hasattr(ox, "features_from_bbox"):
                gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=osm_tags)
            else:
                gdf = ox.geometries_from_bbox(north, south, east, west, tags=osm_tags)

            if gdf is None or gdf.empty:
                return None

            return gdf.to_crs(grid_info["crs"])

        except Exception as e:
            print(f"osm query warning: {e}")
            return None

    def _rasterize_geometries(self, gdf_utm, shape: tuple, transform, value: int, fill: int = 0) -> np.ndarray:
        """rasterizes valid geometries into a mask."""

        if gdf_utm is None or gdf_utm.empty:
            return np.full(shape, fill, dtype=np.uint8)

        shapes = [
            (geom, value)
            for geom in gdf_utm.geometry
            if geom is not None and not geom.is_empty
        ]

        if not shapes:
            return np.full(shape, fill, dtype=np.uint8)

        return rasterize(shapes, out_shape=shape, transform=transform, fill=fill, dtype=np.uint8)

    def _compute_buffered_edt(self, gdf_utm, grid_info: dict, buffer_meters: float = 5000.0) -> np.ndarray:
        """computes euclidean distance in meters on a buffered grid."""

        _, buf_shape, buf_transform = self._get_buffered_grid(grid_info, buffer_meters, 10.0)

        if gdf_utm is None or gdf_utm.empty:
            return np.full(grid_info["shape"], 50000.0, dtype=np.float32)

        mask = self._rasterize_geometries(gdf_utm, buf_shape, buf_transform, value=1, fill=0)

        distance = distance_transform_edt(1 - mask) * 10.0

        offset = int(round(buffer_meters / 10.0))

        result = distance[
            offset:offset + GRID_SIZE_PX,
            offset:offset + GRID_SIZE_PX
        ]

        return result.astype(np.float32)

    def _compute_buffered_accessibility(self, dem_buffered: np.ndarray, passable_buffered: np.ndarray, gdf_utm_buffered, grid_info: dict, buffer_meters: float = 5000.0, max_time_cap_hours: float = 12.0) -> np.ndarray:
        """computes minimum walking time on a buffered terrain graph."""

        buf_shape = dem_buffered.shape

        _, expected_shape, buf_transform = self._get_buffered_grid(grid_info, buffer_meters, 10.0)

        if buf_shape != expected_shape:
            raise ValueError(f"buffered DEM shape {buf_shape} does not match expected shape {expected_shape}")

        sources = self._rasterize_geometries(gdf_utm_buffered, buf_shape, buf_transform, value=1, fill=0)

        sources[passable_buffered == 0] = 0

        if not np.any(sources == 1):
            return np.full(grid_info["shape"], max_time_cap_hours, dtype=np.float32)

        optimal_time = dijkstra_kernel(dem_buffered, sources, passable_buffered, resolution=10.0)

        offset = int(round(buffer_meters / 10.0))

        master_time = optimal_time[
            offset:offset + GRID_SIZE_PX,
            offset:offset + GRID_SIZE_PX
        ]

        clean_time = np.where(np.isinf(master_time), max_time_cap_hours, master_time)

        return np.clip(clean_time, 0.0, max_time_cap_hours).astype(np.float32)

    def fetch_all_spatial_features(self, lat: float, lon: float, scl_10m: np.ndarray = None, buffer_meters: float = 5000.0) -> dict:
        """fetches all static spatial features for one scene."""

        grid_info = self.aligner.get_master_grid_info(lat, lon)

        shape = grid_info["shape"]
        transform = grid_info["transform"]

        terrain = self.fetch_dem_features(grid_info)
        elevation_10m = terrain["Elevation"]

        dem_buffered, _ = self._fetch_dem_raster(grid_info, buffer_meters=buffer_meters, resolution=10.0)

        gdf_roads = self._query_osm_buffered(grid_info, self.tags_roads, buffer_meters)
        gdf_trails = self._query_osm_buffered(grid_info, self.tags_trails, buffer_meters)
        gdf_water = self._query_osm_buffered(grid_info, self.tags_water, buffer_meters)
        gdf_bridges = self._query_osm_buffered(grid_info, self.tags_bridges, buffer_meters)
        gdf_railways = self._query_osm_buffered(grid_info, self.tags_railways, buffer_meters)
        gdf_camps = self._query_osm_buffered(grid_info, self.tags_camps, buffer_meters)
        gdf_power = self._query_osm_buffered(grid_info, self.tags_power, buffer_meters)

        _, buf_shape, buf_transform = self._get_buffered_grid(grid_info, buffer_meters, 10.0)

        passable_buffered = np.ones(buf_shape, dtype=np.uint8)

        water_mask = self._rasterize_geometries(gdf_water, buf_shape, buf_transform, value=1, fill=0)

        passable_buffered[water_mask == 1] = 0

        if scl_10m is not None:
            if scl_10m.shape != shape:
                raise ValueError("scl_10m must have the same shape as the master grid")

            offset = int(round(buffer_meters / 10.0))

            inner_slice = (
                slice(offset, offset + GRID_SIZE_PX),
                slice(offset, offset + GRID_SIZE_PX)
            )

            passable_inner = passable_buffered[inner_slice]
            passable_inner[scl_10m == 6] = 0
            passable_buffered[inner_slice] = passable_inner

        passable_buffered[np.isnan(dem_buffered)] = 0

        bridge_mask = self._rasterize_geometries(gdf_bridges, buf_shape, buf_transform, value=1, fill=0)

        passable_buffered[bridge_mask == 1] = 1

        travel_roads = self._compute_buffered_accessibility(dem_buffered, passable_buffered, gdf_roads, grid_info, buffer_meters)
        travel_trails = self._compute_buffered_accessibility(dem_buffered, passable_buffered, gdf_trails, grid_info, buffer_meters)
        dist_railways = self._compute_buffered_edt(gdf_railways, grid_info, buffer_meters)
        dist_camps = self._compute_buffered_edt(gdf_camps, grid_info, buffer_meters)
        dist_power = self._compute_buffered_edt(gdf_power, grid_info, buffer_meters)

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

    data = fetcher.fetch_all_spatial_features(lat=53.5461, lon=-113.4937)

    for name, arr in data.items():
        print(
            f"{name:20s} shape: {arr.shape}"
            f"range: [{np.nanmin(arr):8.2f}, {np.nanmax(arr):8.2f}]"
        )