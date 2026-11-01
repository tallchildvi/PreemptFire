from datetime import date
import time
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import matplotlib.pyplot as plt
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

from src.config import MIN_DEM_VALID_RATIO
from src.data_pipeline.accessibility.solver import dijkstra_kernel
from src.data_pipeline.osm_temporal_cache import make_temporal_osm_query
from src.processing.grid_aligner import GridAligner


class SpatialFeatureFetcher:
    """COG-optimized DEM and OSM feature extractor with strict DEM/Road quality validation."""

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
        osm_mode: str = "local_history",
        timeout_sec: int = 25,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        allowed_countries: Optional[List[str]] = None,
        allowed_regions: Optional[List[str]] = None,
    ):
        self.osm_mode = osm_mode
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.aligner = GridAligner()
        self.stac_client = self._init_stac_client()

        if self.osm_mode == "local_history":
            from src.data_pipeline.osm_extractor import HistoricalOSMExtractor
            self.osm_extractor = HistoricalOSMExtractor(
                allowed_countries=allowed_countries,
                allowed_regions=allowed_regions,
            )
            self._query_osm_temporal = None
        else:
            self.osm_extractor = None
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
        """Streams only the BBox window from Copernicus DEM COGs and reprojects to target UTM grid."""
        _, target_shape, target_transform = self._get_buffered_grid(grid_info, buffer_meters, resolution)

        if not self._ensure_stac_client():
            raise RuntimeError("STAC client is unavailable; cannot fetch mandatory Copernicus DEM.")

        bbox = (
            tuple(grid_info["bbox_wgs84"])
            if buffer_meters == 0.0
            else self._get_buffered_wgs84_bbox(grid_info, buffer_meters)
        )
        # Add small 0.01 deg margin to prevent edge clipping during UTM reprojection
        padded_bbox = (bbox[0] - 0.01, bbox[1] - 0.01, bbox[2] + 0.01, bbox[3] + 0.01)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                search = self.stac_client.search(collections=["cop-dem-glo-30"], bbox=padded_bbox)
                items = list(search.items())
                if not items:
                    raise RuntimeError(f"No Copernicus DEM tiles returned for bbox: {padded_bbox}")

                src_files = [rasterio.open(pc.sign(item.assets["data"].href)) for item in items]
                try:
                    # Windowed merge: reads ONLY the padded_bbox window from remote COGs
                    mosaic_arr, mosaic_transform = merge(src_files, bounds=padded_bbox)
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
                last_err = e
                wait = self.retry_backoff ** attempt
                print(f"  [DEM] Attempt {attempt + 1}/{self.max_retries} failed ({e}), retrying in {wait:.0f}s...")
                time.sleep(wait)
                self.stac_client = self._init_stac_client()

        raise RuntimeError(f"DEM fetch failed after {self.max_retries} retries: {last_err}")

    def _compute_terrain_from_buffered_dem(
        self,
        dem_buf: np.ndarray,
        buf_transform: rasterio.Affine,
        grid_info: dict,
    ) -> Dict[str, np.ndarray]:
        """Computes slope and aspect on the 50m DEM and aggregates onto the 100m master grid."""
        px, py = abs(buf_transform.a), abs(buf_transform.e)
        dz_dy, dz_dx = np.gradient(dem_buf, py, px)
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
            "Elevation": dem_buf,
            "Slope": slope_deg,
            "Northness": northness,
            "Eastness": eastness,
        }

        aligned = {}
        for name, arr in terrain_layers.items():
            aligned[name] = self.aligner.align_raster_to_master(
                src_array=arr,
                src_crs=grid_info["crs"],
                src_transform=buf_transform,
                grid_info=grid_info,
                resampling_method=Resampling.average,
                dst_nodata=np.nan,
            )
        return aligned

    def _fetch_osm_geometries(
        self, grid_info: dict, target_date: str, buffer_meters: float = 0.0
    ) -> Optional[gpd.GeoDataFrame]:
        """Unified router between local history dumps and live Overpass API."""
        if self.osm_mode == "local_history":
            west, south, east, north = self._get_buffered_wgs84_bbox(
                grid_info, buffer_meters=buffer_meters
            )
            return self.osm_extractor.extract_features_for_date(
                date=target_date,
                west=west,
                south=south,
                east=east,
                north=north,
                target_crs=grid_info["crs"],
            )
        elif self.osm_mode == "overpass":
            return self._query_osm_temporal(
                grid_info, target_date=target_date, buffer_meters=buffer_meters
            )
        return None

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

        shapes = [
            (geom, value)
            for geom in gdf_utm.geometry
            if geom is not None and not geom.is_empty
        ]
        if not shapes:
            return np.full(shape, fill, dtype=np.uint8)
        return rasterize(
            shapes, out_shape=shape, transform=transform, fill=fill, dtype=np.uint8, all_touched=True
        )

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
        shape_50m = dem_50m.shape
        sources = self._rasterize_geometries(gdf_subset, shape_50m, buf_transform_50m, value=1, fill=0)
        sources[passable_50m == 0] = 0

        if not np.any(sources == 1):
            return fallback

        time_50m = dijkstra_kernel(dem_50m, sources, passable_50m, resolution=50.0)
        time_50m = np.where(np.isinf(time_50m), max_time_cap_hours, time_50m)
        time_50m = np.clip(time_50m, 0.0, max_time_cap_hours).astype(np.float32)

        master_time = np.full(grid_info["shape"], max_time_cap_hours, dtype=np.float32)
        reproject(
            source=time_50m,
            destination=master_time,
            src_transform=buf_transform_50m,
            src_crs=grid_info["crs"],
            dst_transform=grid_info["transform"],
            dst_crs=grid_info["crs"],
            resampling=Resampling.average,
            dst_nodata=max_time_cap_hours,
        )
        return master_time

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

        mask_50m = self._rasterize_geometries(gdf_subset, buf_shape_50m, buf_transform_50m, value=1, fill=0)
        if not np.any(mask_50m == 1):
            return fallback

        dist_50m = (distance_transform_edt(1 - mask_50m) * 50.0).astype(np.float32)
        dist_master = np.full(grid_info["shape"], max_dist_m, dtype=np.float32)
        reproject(
            source=dist_50m,
            destination=dist_master,
            src_transform=buf_transform_50m,
            src_crs=grid_info["crs"],
            dst_transform=grid_info["transform"],
            dst_crs=grid_info["crs"],
            resampling=Resampling.average,
            dst_nodata=max_dist_m,
        )
        return dist_master

    @staticmethod
    def _validate_dem_and_roads(features: Dict[str, np.ndarray], expected_shape: Tuple[int, int]) -> None:
        """Strictly checks that DEM layers and Road accessibility are valid and finite."""
        for dem_key in ("Elevation", "Slope", "Northness", "Eastness"):
            arr = features.get(dem_key)
            if arr is None or arr.shape != expected_shape:
                raise RuntimeError(f"Missing or invalid shape for critical DEM layer: {dem_key}")
            finite_ratio = float(np.isfinite(arr).mean())
            if finite_ratio < MIN_DEM_VALID_RATIO:
                raise RuntimeError(
                    f"Critical DEM layer '{dem_key}' has insufficient finite pixels "
                    f"({finite_ratio:.3f} < {MIN_DEM_VALID_RATIO:.3f})."
                )

        roads = features.get("Travel_Time_Roads")
        if roads is None or roads.shape != expected_shape or not np.all(np.isfinite(roads)):
            raise RuntimeError("Critical 'Travel_Time_Roads' layer is missing or contains non-finite values.")

    def fetch_all_spatial_features(
        self,
        lat: float,
        lon: float,
        target_date: Optional[str] = None,
        scl_10m: Optional[np.ndarray] = None,
        buffer_meters: float = 4000.0,
    ) -> Dict[str, np.ndarray]:
        """Fetches DEM terrain derivatives and OSM accessibility/distance fields in a single DEM pass."""
        date_str = target_date.split(" ")[0] if target_date else date.today().strftime("%Y-%m-%d")
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        # Single buffered 50m DEM COG window fetch serves both terrain derivatives and Dijkstra routing
        dem_50m, buf_transform_50m = self._fetch_dem_raster(
            grid_info, buffer_meters=buffer_meters, resolution=50.0
        )
        buf_shape_50m = dem_50m.shape

        terrain = self._compute_terrain_from_buffered_dem(dem_50m, buf_transform_50m, grid_info)

        gdf_all = self._fetch_osm_geometries(grid_info, target_date=date_str, buffer_meters=buffer_meters)

        gdf_roads = gdf_trails = gdf_water = gdf_bridges = None
        gdf_railways = gdf_camps = gdf_power = None

        if gdf_all is not None and not gdf_all.empty:
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

        passable_50m = np.ones(buf_shape_50m, dtype=np.uint8)
        water_mask_50m = self._rasterize_geometries(gdf_water, buf_shape_50m, buf_transform_50m)
        passable_50m[water_mask_50m == 1] = 0
        passable_50m[np.isnan(dem_50m)] = 0

        bridge_mask_50m = self._rasterize_geometries(gdf_bridges, buf_shape_50m, buf_transform_50m)
        passable_50m[bridge_mask_50m == 1] = 1

        travel_roads = self._compute_fast_accessibility_50m(
            dem_50m, passable_50m, gdf_roads, grid_info, buf_transform_50m
        )
        travel_trails = self._compute_fast_accessibility_50m(
            dem_50m, passable_50m, gdf_trails, grid_info, buf_transform_50m
        )
        dist_railways = self._compute_fast_edt_50m(gdf_railways, grid_info, buf_shape_50m, buf_transform_50m)
        dist_camps = self._compute_fast_edt_50m(gdf_camps, grid_info, buf_shape_50m, buf_transform_50m)
        dist_power = self._compute_fast_edt_50m(gdf_power, grid_info, buf_shape_50m, buf_transform_50m)

        results = {
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

        self._validate_dem_and_roads(results, grid_info["shape"])
        return results


if __name__ == "__main__":
    fetcher = SpatialFeatureFetcher(osm_mode="local_history")
    test_lat, test_lon, test_date = 41.6986, -7.0011, "2022-10-09"

    print(f"Testing SpatialFeatureFetcher (100m grid) for [{test_lat}, {test_lon}] on {test_date}...")
    t_start = time.perf_counter()
    data = fetcher.fetch_all_spatial_features(lat=test_lat, lon=test_lon, target_date=test_date)
    elapsed = time.perf_counter() - t_start

    print(f"Completed in {elapsed:.2f}s:")
    for name, arr in data.items():
        print(f"  {name:<22} shape: {arr.shape} | range: [{np.nanmin(arr):8.2f}, {np.nanmax(arr):8.2f}]")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for ax, (name, arr) in zip(axes.ravel(), data.items()):
        cmap = "terrain" if name in ("Elevation", "Slope") else ("coolwarm" if "ness" in name else "magma_r")
        im = ax.imshow(arr, cmap=cmap, origin="upper")
        ax.set_title(f"{name}\n[{np.nanmin(arr):.1f}, {np.nanmax(arr):.1f}]", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"SpatialFeatureFetcher 100m Verification [{test_lat}, {test_lon}]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()