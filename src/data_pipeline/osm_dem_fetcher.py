from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import osmnx as ox
import planetary_computer as pc
import pystac_client
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

from src.config import GRID_SIZE_PX
from src.data_pipeline.accessibility.solver import dijkstra_kernel
from src.processing.grid_aligner import GridAligner


class SpatialFeatureFetcher:
    """High-reliability spatial feature fetcher with multi-endpoint Overpass routing."""

    OVERPASS_SERVERS = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]

    TARGETED_OSM_TAGS = {
        "highway": [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "unclassified", "residential", "service", "track", "path", "footway"
        ],
        "railway": ["rail", "narrow_gauge", "spur"],
        "power": ["line", "minor_line", "cable", "substation"],
        "natural": ["water", "wetland"],
        "waterway": ["river", "stream", "canal"],
        "tourism": ["camp_site", "picnic_site", "wilderness_hut"],
        "amenity": ["shelter", "firepit"],
    }

    ROAD_HIGHWAYS = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service"
    }
    TRAIL_HIGHWAYS = {"track", "path", "footway"}

    def __init__(self, timeout_sec: int = 25):
        self.timeout_sec = timeout_sec
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        self.aligner = GridAligner()

        # Configure OSMnx resilience settings
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.timeout = 10
    
        ox.settings.overpass_endpoint = self.OVERPASS_SERVERS[0]

    def _get_buffered_grid(self, grid_info: dict, buffer_meters: float, resolution: float) -> Tuple:
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        b_min_x, b_max_x = min_x - buffer_meters, max_x + buffer_meters
        b_min_y, b_max_y = min_y - buffer_meters, max_y + buffer_meters

        width = int(round((b_max_x - b_min_x) / resolution))
        height = int(round((b_max_y - b_min_y) / resolution))
        transform = from_bounds(b_min_x, b_min_y, b_max_x, b_max_y, width, height)

        return (b_min_x, b_min_y, b_max_x, b_max_y), (height, width), transform

    def _get_buffered_wgs84_bbox(self, grid_info: dict, buffer_meters: float) -> Tuple[float, float, float, float]:
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        transformer = Transformer.from_crs(grid_info["crs"], "EPSG:4326", always_xy=True)
        corners = [
            (min_x - buffer_meters, min_y - buffer_meters),
            (min_x - buffer_meters, max_y + buffer_meters),
            (max_x + buffer_meters, min_y - buffer_meters),
            (max_x + buffer_meters, max_y + buffer_meters),
        ]
        corners_wgs84 = [transformer.transform(x, y) for x, y in corners]
        lons = [p[0] for p in corners_wgs84]
        lats = [p[1] for p in corners_wgs84]
        return min(lons), min(lats), max(lons), max(lats)

    def _fetch_dem_raster(self, grid_info: dict, buffer_meters: float, resolution: float) -> Tuple[np.ndarray, rasterio.Affine]:
        bbox = grid_info["bbox_wgs84"] if buffer_meters == 0 else self._get_buffered_wgs84_bbox(grid_info, buffer_meters)
        search = self.stac_client.search(collections=["cop-dem-glo-30"], bbox=bbox)
        items = list(search.items())
        if not items:
            raise RuntimeError(f"No Copernicus DEM items found for bbox: {bbox}")

        src_files = [rasterio.open(item.assets["data"].href) for item in items]
        try:
            mosaic_arr, mosaic_transform = merge(src_files)
            src_crs = src_files[0].crs
            src_nodata = src_files[0].nodata
        finally:
            for src in src_files:
                src.close()

        _, target_shape, target_transform = self._get_buffered_grid(grid_info, buffer_meters, resolution)
        dem = np.full(target_shape, np.nan, dtype=np.float32)

        reproject(
            source=mosaic_arr[0],
            destination=dem,
            src_transform=mosaic_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=grid_info["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
        return dem, target_transform

    def fetch_dem_features(self, grid_info: dict) -> Dict[str, np.ndarray]:
        dem_30m, native_transform = self._fetch_dem_raster(grid_info, buffer_meters=0.0, resolution=30.0)

        px, py = abs(native_transform.a), abs(native_transform.e)
        dz_dy, dz_dx = np.gradient(dem_30m, py, px)
        dz_dnorth = -dz_dy

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dnorth**2))
        slope_deg = np.degrees(slope_rad).astype(np.float32)

        aspect_rad = np.arctan2(-dz_dx, -dz_dnorth) % (2.0 * np.pi)
        northness = np.cos(aspect_rad).astype(np.float32)
        eastness = np.sin(aspect_rad).astype(np.float32)

        flat_mask = slope_deg < 0.1
        northness[flat_mask] = 0.0
        eastness[flat_mask] = 0.0

        terrain_layers = {
            "Elevation": dem_30m,
            "Slope": slope_deg,
            "Northness": northness,
            "Eastness": eastness,
        }

        aligned = {}
        for name, arr in terrain_layers.items():
            aligned[name] = self.aligner.align_raster_to_master(
                src_array=arr,
                src_crs=grid_info["crs"],
                src_transform=native_transform,
                grid_info=grid_info,
                resampling_method=Resampling.bilinear,
                dst_nodata=np.nan,
            )
        return aligned

    def _query_osm_resilient(self, grid_info: dict, buffer_meters: float = 4000.0) -> Optional[gpd.GeoDataFrame]:
        """Queries Overpass with automatic multi-mirror failover and fast timeout."""
        west, south, east, north = self._get_buffered_wgs84_bbox(grid_info, buffer_meters)

        for endpoint in self.OVERPASS_SERVERS:
            ox.settings.overpass_endpoint = endpoint
            try:
                if hasattr(ox, "features_from_bbox"):
                    gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=self.TARGETED_OSM_TAGS)
                else:
                    gdf = ox.geometries_from_bbox(north, south, east, west, tags=self.TARGETED_OSM_TAGS)

                if gdf is not None and not gdf.empty:
                    return gdf.to_crs(grid_info["crs"])
                return None
            except Exception:
                continue

        print("  [OSM Warning] All Overpass endpoints timed out. Using default distance matrices.")
        return None

    def _rasterize_geometries(self, gdf_utm, shape: Tuple[int, int], transform, value: int = 1, fill: int = 0) -> np.ndarray:
        if gdf_utm is None or gdf_utm.empty:
            return np.full(shape, fill, dtype=np.uint8)
        shapes = [(geom, value) for geom in gdf_utm.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            return np.full(shape, fill, dtype=np.uint8)
        return rasterize(shapes, out_shape=shape, transform=transform, fill=fill, dtype=np.uint8)

    def _compute_fast_accessibility_50m(
        self,
        dem_50m: np.ndarray,
        passable_50m: np.ndarray,
        gdf_subset,
        grid_info: dict,
        buf_transform_50m,
        max_time_cap_hours: float = 12.0,
    ) -> np.ndarray:
        shape_50m = dem_50m.shape
        sources = self._rasterize_geometries(gdf_subset, shape_50m, buf_transform_50m, value=1, fill=0)
        sources[passable_50m == 0] = 0

        if not np.any(sources == 1):
            return np.full(grid_info["shape"], max_time_cap_hours, dtype=np.float32)

        time_50m = dijkstra_kernel(dem_50m, sources, passable_50m, resolution=50.0)
        time_50m = np.where(np.isinf(time_50m), max_time_cap_hours, time_50m)
        time_50m = np.clip(time_50m, 0.0, max_time_cap_hours).astype(np.float32)

        master_time_10m = np.full(grid_info["shape"], max_time_cap_hours, dtype=np.float32)
        reproject(
            source=time_50m,
            destination=master_time_10m,
            src_transform=buf_transform_50m,
            src_crs=grid_info["crs"],
            dst_transform=grid_info["transform"],
            dst_crs=grid_info["crs"],
            resampling=Resampling.bilinear,
            dst_nodata=max_time_cap_hours,
        )
        return master_time_10m

    def _compute_fast_edt_50m(
        self, gdf_subset, grid_info: dict, buf_shape_50m: tuple, buf_transform_50m, max_dist_m: float = 50000.0
    ) -> np.ndarray:
        if gdf_subset is None or gdf_subset.empty:
            return np.full(grid_info["shape"], max_dist_m, dtype=np.float32)

        mask_50m = self._rasterize_geometries(gdf_subset, buf_shape_50m, buf_transform_50m, value=1, fill=0)
        dist_50m = (distance_transform_edt(1 - mask_50m) * 50.0).astype(np.float32)

        dist_10m = np.full(grid_info["shape"], max_dist_m, dtype=np.float32)
        reproject(
            source=dist_50m,
            destination=dist_10m,
            src_transform=buf_transform_50m,
            src_crs=grid_info["crs"],
            dst_transform=grid_info["transform"],
            dst_crs=grid_info["crs"],
            resampling=Resampling.bilinear,
            dst_nodata=max_dist_m,
        )
        return dist_10m

    def fetch_all_spatial_features(self, lat: float, lon: float, scl_10m: np.ndarray = None, buffer_meters: float = 4000.0) -> Dict[str, np.ndarray]:
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        # 1. 10m Terrain layers
        terrain = self.fetch_dem_features(grid_info)

        # 2. 50m DEM for graph routing
        dem_50m, buf_transform_50m = self._fetch_dem_raster(grid_info, buffer_meters=buffer_meters, resolution=50.0)
        _, buf_shape_50m, _ = self._get_buffered_grid(grid_info, buffer_meters, 50.0)

        # 3. Resilient OSM Query with server failover
        gdf_all = self._query_osm_resilient(grid_info, buffer_meters=buffer_meters)

        gdf_roads, gdf_trails, gdf_water = None, None, None
        gdf_railways, gdf_camps, gdf_power = None, None, None

        if gdf_all is not None and not gdf_all.empty:
            if "highway" in gdf_all.columns:
                gdf_roads = gdf_all[gdf_all["highway"].isin(self.ROAD_HIGHWAYS)]
                gdf_trails = gdf_all[gdf_all["highway"].isin(self.TRAIL_HIGHWAYS)]
            if "natural" in gdf_all.columns or "waterway" in gdf_all.columns:
                w_cond = False
                if "natural" in gdf_all.columns:
                    w_cond = gdf_all["natural"].isin(["water", "wetland"])
                if "waterway" in gdf_all.columns:
                    w_cond = w_cond | gdf_all["waterway"].isin(["river", "stream", "canal"])
                gdf_water = gdf_all[w_cond]
            if "railway" in gdf_all.columns:
                gdf_railways = gdf_all[gdf_all["railway"].isin(self.TARGETED_OSM_TAGS["railway"])]
            if "power" in gdf_all.columns:
                gdf_power = gdf_all[gdf_all["power"].isin(self.TARGETED_OSM_TAGS["power"])]
            if "tourism" in gdf_all.columns or "amenity" in gdf_all.columns:
                c_cond = False
                if "tourism" in gdf_all.columns:
                    c_cond = gdf_all["tourism"].isin(self.TARGETED_OSM_TAGS["tourism"])
                if "amenity" in gdf_all.columns:
                    c_cond = c_cond | gdf_all["amenity"].isin(self.TARGETED_OSM_TAGS["amenity"])
                gdf_camps = gdf_all[c_cond]

        # 4. Passable matrix
        passable_50m = np.ones(buf_shape_50m, dtype=np.uint8)
        water_mask_50m = self._rasterize_geometries(gdf_water, buf_shape_50m, buf_transform_50m, value=1, fill=0)
        passable_50m[water_mask_50m == 1] = 0
        passable_50m[np.isnan(dem_50m)] = 0

        # 5. Fast Distance & Travel Time matrices
        travel_roads = self._compute_fast_accessibility_50m(dem_50m, passable_50m, gdf_roads, grid_info, buf_transform_50m)
        travel_trails = self._compute_fast_accessibility_50m(dem_50m, passable_50m, gdf_trails, grid_info, buf_transform_50m)
        dist_railways = self._compute_fast_edt_50m(gdf_railways, grid_info, buf_shape_50m, buf_transform_50m)
        dist_camps = self._compute_fast_edt_50m(gdf_camps, grid_info, buf_shape_50m, buf_transform_50m)
        dist_power = self._compute_fast_edt_50m(gdf_power, grid_info, buf_shape_50m, buf_transform_50m)

        return {
            "Elevation": terrain["Elevation"],
            "Slope": terrain["Slope"],
            "Northness": terrain["Northness"],
            "Eastness": terrain["Eastness"],
            "Travel_Time_Roads": travel_roads,
            "Travel_Time_Trails": travel_trails,
            "Dist_to_Railways": dist_railways,
            "Dist_to_Camps": dist_camps,
            "Dist_to_Powerlines": dist_power,
        }

    
if __name__ == "__main__":
    fetcher = SpatialFeatureFetcher()

    data = fetcher.fetch_all_spatial_features(lat=53.5461, lon=-113.4937)

    for name, arr in data.items():
        print(
            f"{name:20s} shape: {arr.shape}"
            f"range: [{np.nanmin(arr):8.2f}, {np.nanmax(arr):8.2f}]"
        )