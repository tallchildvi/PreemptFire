from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from src.config import MAX_INVALID_RATIO, MIN_DEM_VALID_RATIO, PATCH_SIZE_PX, STRIDE_PX
from src.data_pipeline.era5_fetcher import ERA5Fetcher
from src.data_pipeline.osm_dem_fetcher import SpatialFeatureFetcher
from src.data_pipeline.population_fetcher import PopulationFetcher
from src.data_pipeline.sentinel_fetcher import SentinelFetcher
from src.data_pipeline.weather_fetcher import WeatherFetcher
from src.processing.grid_aligner import GridAligner


class SingleSceneCollector:
    """Orchestrates raw 100m single-patch geospatial and meteorological data collection without indices or targets."""

    REQUIRED_S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    REQUIRED_DEM_LAYERS = ("Elevation", "Slope", "Northness", "Eastness")

    def __init__(self, osm_mode: str = "local_history", include_cffdrs: bool = False):
        self.aligner = GridAligner()
        self.sentinel_fetcher = SentinelFetcher(max_nodata_fraction=MAX_INVALID_RATIO)
        self.spatial_fetcher = SpatialFeatureFetcher(osm_mode=osm_mode)
        self.population_fetcher = PopulationFetcher()
        self.era5_fetcher = ERA5Fetcher()
        self.weather_fetcher = WeatherFetcher()
        self.include_cffdrs = include_cffdrs
        if self.include_cffdrs:
            from src.data_pipeline.cffdrs_fetcher import CFFDRSFetcher
            self.cffdrs_fetcher = CFFDRSFetcher()
        else:
            self.cffdrs_fetcher = None

    def _validate_collected_scene(
        self,
        rasters_2d: Dict[str, np.ndarray],
        context_1d: Dict[str, float],
        loss_mask: np.ndarray,
        expected_shape: tuple[int, int],
    ) -> tuple[bool, str]:
        """Hard quality gate ensuring raw Sentinel-2, DEM, Road accessibility, and 1D context are valid."""
        invalid_ratio = float((loss_mask > 0.5).mean())
        if invalid_ratio > MAX_INVALID_RATIO:
            return False, f"Scene invalid pixel ratio {invalid_ratio:.3f} exceeds limit {MAX_INVALID_RATIO:.3f}"

        for band in self.REQUIRED_S2_BANDS:
            arr = rasters_2d.get(band)
            if arr is None or arr.shape != expected_shape:
                return False, f"Missing or misshaped Sentinel-2 band: {band}"
            if float(np.isfinite(arr).mean()) < (1.0 - MAX_INVALID_RATIO):
                return False, f"Sentinel-2 band {band} has excessive NaN values"

        for dem_name in self.REQUIRED_DEM_LAYERS:
            arr = rasters_2d.get(dem_name)
            if arr is None or arr.shape != expected_shape:
                return False, f"Missing or misshaped DEM layer: {dem_name}"
            if float(np.isfinite(arr).mean()) < MIN_DEM_VALID_RATIO:
                return False, f"DEM layer {dem_name} failed finite ratio check (< {MIN_DEM_VALID_RATIO})"

        roads = rasters_2d.get("Travel_Time_Roads")
        if roads is None or roads.shape != expected_shape or not np.all(np.isfinite(roads)):
            return False, "Missing or non-finite Travel_Time_Roads layer"

        if not context_1d:
            return False, "Empty 1D context feature dictionary"
        for k, v in context_1d.items():
            if v is None or not np.isfinite(float(v)):
                return False, f"Non-finite 1D context feature: {k}={v}"

        return True, "OK"

    def collect_sample(
        self,
        lat: float,
        lon: float,
        target_date: str,
        is_fire: int | bool = 1,
    ) -> Optional[Dict[str, Any]]:
        total_start = time.perf_counter()
        timings: Dict[str, float] = {}

        grid_info = self.aligner.get_master_grid_info(lat, lon)
        center_row, center_col = self.aligner.latlon_to_pixel(lat, lon, grid_info)
        target_year = datetime.strptime(target_date, "%Y-%m-%d").year

        pre_fire_dt = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)
        pre_fire_date = pre_fire_dt.strftime("%Y-%m-%d")

        # Stage 1: Fetch raw Sentinel-2 T0 and Sentinel-1 SAR
        t_s2 = time.perf_counter()
        sentinel_data = self.sentinel_fetcher.fetch_all_radar_optical(lat, lon, target_date)
        timings["sentinel_fetch"] = time.perf_counter() - t_s2

        if not sentinel_data:
            print("  [SingleSceneCollector] Rejected: No valid cloud-free Sentinel-2 T0 scene found.")
            return None

        bands_t0 = sentinel_data.get("bands_t0")
        t0_date = sentinel_data.get("t0_date")
        if not bands_t0 or not t0_date:
            return None

        loss_mask = sentinel_data.get("loss_mask")
        sar_bands = sentinel_data.get("sar_bands") or {}

        # Stage 2: Concurrent extraction of DEM/OSM, Population, Soil Moisture, and Weather
        def _fetch_dem_osm():
            t0 = time.perf_counter()
            res = self.spatial_fetcher.fetch_all_spatial_features(
                lat=lat, lon=lon, target_date=target_date, scl_10m=None
            )
            timings["dem_osm_fetch"] = time.perf_counter() - t0
            return res

        def _fetch_population_1d():
            t0 = time.perf_counter()
            res = self.population_fetcher.fetch_population_1d(
                lat=lat, lon=lon, year=min(target_year, 2020)
            )
            timings["population_gee_1d"] = time.perf_counter() - t0
            return res

        def _fetch_soil():
            t0 = time.perf_counter()
            res = self.era5_fetcher.fetch_soil_moisture(
                grid_info=grid_info, target_date=pre_fire_date
            )
            timings["soil_moisture_gee"] = time.perf_counter() - t0
            return res

        def _fetch_weather():
            t0 = time.perf_counter()
            res = self.weather_fetcher.fetch_all_weather_metrics(
                lat=lat, lon=lon, target_date=pre_fire_date, t0_date=t0_date
            )
            timings["weather_all"] = time.perf_counter() - t0
            return res

        def _fetch_cffdrs():
            if not self.include_cffdrs or self.cffdrs_fetcher is None:
                return {}
            t0 = time.perf_counter()
            res = self.cffdrs_fetcher.fetch_cffdrs_metrics(
                lat=lat, lon=lon, date_t0=pre_fire_date
            )
            timings["cffdrs_calc"] = time.perf_counter() - t0
            return res

        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                f_spatial = pool.submit(_fetch_dem_osm)
                f_population = pool.submit(_fetch_population_1d)
                f_soil = pool.submit(_fetch_soil)
                f_weather = pool.submit(_fetch_weather)
                f_cffdrs = pool.submit(_fetch_cffdrs)

                spatial_features = f_spatial.result()
                pop_context_1d = f_population.result()
                soil_moisture = f_soil.result()
                weather_metrics = f_weather.result()
                cffdrs_metrics = f_cffdrs.result()
        except Exception as e:
            print(f"  [SingleSceneCollector] Rejected due to sub-module failure: {e}")
            return None

        raw_rasters: Dict[str, Any] = {
            **{k: v for k, v in bands_t0.items() if k != "SCL"},
            **sar_bands,
            **spatial_features,
            "Soil_Moisture": soil_moisture,
        }
        rasters_2d = {
            k: v for k, v in raw_rasters.items() if isinstance(v, np.ndarray) and v.ndim == 2
        }

        context_1d = {
            **weather_metrics,
            **cffdrs_metrics,
            **pop_context_1d,
        }

        # Stage 3: Hard Quality Gate Validation
        is_valid, reason = self._validate_collected_scene(
            rasters_2d=rasters_2d,
            context_1d=context_1d,
            loss_mask=loss_mask,
            expected_shape=grid_info["shape"],
        )
        if not is_valid:
            print(f"  [SingleSceneCollector] Rejected by quality gate: {reason}")
            return None

        total_elapsed = time.perf_counter() - total_start
        timings["total_pipeline_time"] = total_elapsed

        return {
            "metadata": {
                "lat": float(lat),
                "lon": float(lon),
                "center_row": round(center_row, 3),
                "center_col": round(center_col, 3),
                "target_date": target_date,
                "pre_fire_date": pre_fire_date,
                "t0_date": t0_date,
                "is_fire": int(is_fire),
                "crs": str(grid_info["crs"]),
                "utm_bounds": list(grid_info["utm_bounds"]),
                "elapsed_seconds": total_elapsed,
                "timings": timings,
            },
            "rasters_2d": rasters_2d,
            "context_1d": context_1d,
            "loss_mask": loss_mask,
            "scl": bands_t0["SCL"].astype(np.uint8),
        }


if __name__ == "__main__":
    from src.data_pipeline.patch_extractor import PatchExtractor

    collector = SingleSceneCollector(osm_mode="local_history", include_cffdrs=False)
    extractor = PatchExtractor(
        patch_size=PATCH_SIZE_PX,
        stride=STRIDE_PX,
        max_invalid_ratio=MAX_INVALID_RATIO,
    )

    test_lat = 41.6986
    test_lon = -7.0011
    test_date = "2022-10-09"

    print(f"Collecting raw 100m scene ({test_lat}, {test_lon}) on {test_date}...")
    sample = collector.collect_sample(
        lat=test_lat, lon=test_lon, target_date=test_date, is_fire=1
    )

    if not sample:
        print("Scene collection failed or was rejected by quality gate.")
        raise SystemExit(1)

    meta = sample["metadata"]
    print(f"Timings (total: {meta['elapsed_seconds']:.2f}s | T0: {meta['t0_date']}):")
    for k, v in meta["timings"].items():
        print(f"  {k:<28}: {v:>8.2f}s")

    context_1d = sample["context_1d"]
    print(f"\n1D context features ({len(context_1d)} features):")
    for key in sorted(context_1d.keys()):
        print(f"  {key:<32}: {context_1d[key]:>12.4f}")

    scene_id = f"SCENE_{test_lat:.4f}_{test_lon:.4f}_{test_date.replace('-', '')}"
    patches = list(extractor.extract_patches(sample, scene_id=scene_id))
    print(f"Extracted {len(patches)} full-scene patch(es) of shape {patches[0]['X_2d'].shape if patches else 'N/A'}.")

    if patches:
        p = patches[0]
        ch_names = p["channel_names_2d"]
        x_2d = p["X_2d"]

        preview_channels = [
            "B02", "B04", "B08", "B11", "B12",
            "Elevation", "Slope", "Northness", "Eastness",
            "Travel_Time_Roads", "Dist_to_Powerlines", "Soil_Moisture",
        ]
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        axes = axes.ravel()

        for idx, ch_name in enumerate(preview_channels):
            ax = axes[idx]
            ch_idx = ch_names.index(ch_name)
            arr = x_2d[ch_idx]
            cmap = "terrain" if ch_name in ("Elevation", "Slope") else "viridis"
            im = ax.imshow(arr, cmap=cmap, origin="upper")
            ax.plot(meta["center_col"], meta["center_row"], marker="x", color="red", markersize=8)
            ax.set_title(f"{ch_name}\n[{arr.min():.2f}, {arr.max():.2f}]", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(
            f"SingleSceneCollector Raw 100m Overview: {scene_id} (22 Raw Channels, {PATCH_SIZE_PX}x{PATCH_SIZE_PX})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()