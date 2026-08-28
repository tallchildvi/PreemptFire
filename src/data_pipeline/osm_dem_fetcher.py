from datetime import date
import time
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

from src.data_pipeline.accessibility.solver import dijkstra_kernel
from src.data_pipeline.osm_temporal_cache import make_temporal_osm_query
from src.processing.grid_aligner import GridAligner


class SpatialFeatureFetcher:
    """Production-grade DEM and OSM feature extractor with temporal disk caching"""

    TARGETED_OSM_TAGS: Dict[str, List[str]] = {
        "highway": [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "unclassified", "residential", "service", "track", "path", "footway",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
        ],
        "railway": ["rail", "narrow_gauge", "spur"],
        "power": ["line", "minor_line", "cable", "substation"],
        "natural": ["water", "wetland"],
        "waterway": ["river", "stream", "canal", "riverbank"],
        "tourism": ["camp_site", "picnic_site", "wilderness_hut"],
        "amenity": ["shelter", "firepit"],
        "bridge": ["yes"],
    }

    ROAD_HIGHWAYS: set[str] = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    }
    TRAIL_HIGHWAYS: set[str] = {"track", "path", "footway"}

    def __init__(
        self,
        timeout_sec: int = 25,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
    ):
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.aligner = GridAligner()
        self.stac_client = self._init_stac_client()
        self._query_osm_temporal = make_temporal_osm_query()

    def _init_stac_client(self) -> Optional[pystac_client.Client]:
        for attempt in range(self.max_retries):
            try:
                return pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=pc.sign_inplace,
                )
            except Exception as e:
                wait = self.retry_backoff ** attempt
                print(f"  [STAC] Init attempt {attempt + 1}/{self.max_retries} failed ({e}), retrying in {wait:.0f}s...")
                time.sleep(wait)

        print("  [STAC] All client init attempts failed — DEM layers will fallback to NaN.")
        return None

    def _ensure_stac_client(self) -> bool:
        if self.stac_client is None:
            self.stac_client = self._init_stac_client()
        return self.stac_client is not None

    def _get_buffered_grid(
        self, grid_info: dict, buffer_meters: float, resolution: float
    ) -> Tuple[Tuple[float, float, float, float], Tuple[int, int], rasterio.Affine]:
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        b_min_x, b_max_x = min_x - buffer_meters, max_x + buffer_meters
        b_min_y, b_max_y = min_y - buffer_meters, max_y + buffer_meters

        width = int(round((b_max_x - b_min_x) / resolution))
        height = int(round((b_max_y - b_min_y) / resolution))
        transform = from_bounds(b_min_x, b_min_y, b_max_x, b_max_y, width, height)

        return (b_min_x, b_min_y, b_max_x, b_max_y), (height, width), transform

    def _get_buffered_wgs84_bbox(
        self, grid_info: dict, buffer_meters: float
    ) -> Tuple[float, float, float, float]:
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

    def _fetch_dem_raster(
        self, grid_info: dict, buffer_meters: float, resolution: float
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        _, target_shape, target_transform = self._get_buffered_grid(grid_info, buffer_meters, resolution)
        fallback = np.full(target_shape, np.nan, dtype=np.float32), target_transform

        if not self._ensure_stac_client():
            return fallback

        bbox = (
            grid_info["bbox_wgs84"]
            if buffer_meters == 0.0
            else self._get_buffered_wgs84_bbox(grid_info, buffer_meters)
        )

        for attempt in range(self.max_retries):
            try:
                search = self.stac_client.search(collections=["cop-dem-glo-30"], bbox=bbox)
                items = list(search.items())
                if not items:
                    print(f"  [DEM] No items returned for bbox: {bbox}")
                    return fallback

                src_files = [rasterio.open(item.assets["data"].href) for item in items]
                try:
                    mosaic_arr, mosaic_transform = merge(src_files)
                    src_crs = src_files[0].crs
                    src_nodata = src_files[0].nodata
                finally:
                    for src in src_files:
                        try:
                            src.close()
                        except Exception:
                            pass

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

            except Exception as e:
                wait = self.retry_backoff ** attempt
                print(f"  [DEM] Attempt {attempt + 1}/{self.max_retries} failed ({e}), retrying in {wait:.0f}s...")
                time.sleep(wait)
                self.stac_client = self._init_stac_client()

        print("  [DEM] All retries exhausted — returning NaN array.")
        return fallback

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
            try:
                aligned[name] = self.aligner.align_raster_to_master(
                    src_array=arr,
                    src_crs=grid_info["crs"],
                    src_transform=native_transform,
                    grid_info=grid_info,
                    resampling_method=Resampling.bilinear,
                    dst_nodata=np.nan,
                )
            except Exception as e:
                print(f"  [Terrain] Alignment failed for {name} ({e}) — filling with NaN.")
                aligned[name] = np.full(grid_info["shape"], np.nan, dtype=np.float32)

        return aligned

    def _rasterize_geometries(
        self,
        gdf_utm: Optional[gpd.GeoDataFrame],
        shape: Tuple[int, int],
        transform: rasterio.Affine,
        value: int = 1,
        fill: int = 0,
    ) -> np.ndarray:
        if gdf_utm is None or gdf_utm.empty:
            return np.full(shape, fill, dtype=np.uint8)

        try:
            shapes = [
                (geom, value)
                for geom in gdf_utm.geometry
                if geom is not None and not geom.is_empty
            ]
            if not shapes:
                return np.full(shape, fill, dtype=np.uint8)
            return rasterize(shapes, out_shape=shape, transform=transform, fill=fill, dtype=np.uint8)
        except Exception as e:
            print(f"  [Rasterize] Error rasterizing geometries ({e}) — using fill={fill}.")
            return np.full(shape, fill, dtype=np.uint8)

    def _compute_fast_accessibility_50m(
        self,
        dem_50m: np.ndarray,
        passable_50m: np.ndarray,
        gdf_subset: Optional[gpd.GeoDataFrame],
        grid_info: dict,
        buf_transform_50m: rasterio.Affine,
        max_time_cap_hours: float = 12.0,
    ) -> np.ndarray:
        fallback = np.full(grid_info["shape"], max_time_cap_hours, dtype=np.float32)
        try:
            shape_50m = dem_50m.shape
            sources = self._rasterize_geometries(gdf_subset, shape_50m, buf_transform_50m, value=1, fill=0)
            sources[passable_50m == 0] = 0

            if not np.any(sources == 1):
                return fallback

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

        except Exception as e:
            print(f"  [Accessibility] Dijkstra kernel failed ({e}) — using fallback cap.")
            return fallback

    def _compute_fast_edt_50m(
        self,
        gdf_subset: Optional[gpd.GeoDataFrame],
        grid_info: dict,
        buf_shape_50m: Tuple[int, int],
        buf_transform_50m: rasterio.Affine,
        max_dist_m: float = 50000.0,
    ) -> np.ndarray:
        fallback = np.full(grid_info["shape"], max_dist_m, dtype=np.float32)
        if gdf_subset is None or gdf_subset.empty:
            return fallback

        try:
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

        except Exception as e:
            print(f"  [EDT] Distance transform failed ({e}) — using fallback cap.")
            return fallback


    def fetch_all_spatial_features(
        self,
        lat: float,
        lon: float,
        target_date: Optional[str] = None,
        scl_10m: Optional[np.ndarray] = None,
        buffer_meters: float = 4000.0,
    ) -> Dict[str, np.ndarray]:
        date_str = target_date.split(" ")[0] if target_date else date.today().strftime("%Y-%m-%d")
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        # 1. High-resolution terrain features
        terrain = self.fetch_dem_features(grid_info)

        # 2. Downscaled 50m buffered DEM for routing
        dem_50m, buf_transform_50m = self._fetch_dem_raster(
            grid_info, buffer_meters=buffer_meters, resolution=50.0
        )
        _, buf_shape_50m, _ = self._get_buffered_grid(grid_info, buffer_meters, 50.0)

        # 3. Disk-cached temporal OSM query
        gdf_all = self._query_osm_temporal(grid_info, target_date=date_str, buffer_meters=buffer_meters)

        gdf_roads = gdf_trails = gdf_water = gdf_bridges = None
        gdf_railways = gdf_camps = gdf_power = None

        if gdf_all is not None and not gdf_all.empty:
            try:
                if "highway" in gdf_all.columns:
                    gdf_roads = gdf_all[gdf_all["highway"].isin(self.ROAD_HIGHWAYS)]
                    gdf_trails = gdf_all[gdf_all["highway"].isin(self.TRAIL_HIGHWAYS)]

                if "natural" in gdf_all.columns or "waterway" in gdf_all.columns:
                    w_cond = False
                    if "natural" in gdf_all.columns:
                        w_cond = gdf_all["natural"].isin(self.TARGETED_OSM_TAGS["natural"])
                    if "waterway" in gdf_all.columns:
                        w_cond = w_cond | gdf_all["waterway"].isin(self.TARGETED_OSM_TAGS["waterway"])
                    gdf_water = gdf_all[w_cond]

                if "bridge" in gdf_all.columns:
                    gdf_bridges = gdf_all[gdf_all["bridge"].notna() & (gdf_all["bridge"] != "no")]

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

            except Exception as e:
                print(f"  [OSM] Feature filtering error ({e}) — assigning null geometries.")

        # 4. Passable terrain grid (water & void DEM cells blocked, bridges unblocked)
        passable_50m = np.ones(buf_shape_50m, dtype=np.uint8)
        try:
            water_mask_50m = self._rasterize_geometries(gdf_water, buf_shape_50m, buf_transform_50m)
            passable_50m[water_mask_50m == 1] = 0
            passable_50m[np.isnan(dem_50m)] = 0

            # Unblock bridges over water bodies
            bridge_mask_50m = self._rasterize_geometries(gdf_bridges, buf_shape_50m, buf_transform_50m)
            passable_50m[bridge_mask_50m == 1] = 1
        except Exception as e:
            print(f"  [Passable] Passability mask building error ({e}) — defaulting to fully passable.")

        # 5. Spatial distance fields & accessibility maps
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

    test_lat, test_lon = 53.5461, -113.4937
    test_date = "2023-06-15"

    print(f"fetching spatial features for [{test_lat}, {test_lon}] on {test_date}...")
    t_start = time.perf_counter()
    data = fetcher.fetch_all_spatial_features(lat=test_lat, lon=test_lon, target_date=test_date)
    elapsed = time.perf_counter() - t_start

    print(f"\ncompleted in {elapsed:.2f}s:")
    for name, arr in data.items():
        print(
            f"  {name:20s} shape: {arr.shape} | "
            f"range: [{np.nanmin(arr):8.2f}, {np.nanmax(arr):8.2f}]"
        )