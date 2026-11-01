import os
from src.config import COARSE, PIXEL_SCALE_METERS
LIGHT = os.getenv("PF_LIGHT", "1" if COARSE else "0") == "1"

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from typing import Any, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.data_pipeline.cffdrs_fetcher import CFFDRSFetcher
from src.data_pipeline.era5_fetcher import ERA5Fetcher
from src.data_pipeline.index_calculator import IndexCalculator
from src.data_pipeline.osm_dem_fetcher import SpatialFeatureFetcher
from src.data_pipeline.population_fetcher import PopulationFetcher
from src.data_pipeline.sentinel_fetcher import SentinelFetcher
from src.data_pipeline.target_builder import TargetBuilder
from src.data_pipeline.weather_fetcher import WeatherFetcher
from src.processing.grid_aligner import GridAligner


class SingleSceneCollector:
    """Orchestrates multi-modal geospatial and multi-interval meteorological data collection."""

    def __init__(self, osm_mode: str = "local_history"):
        print(f"[SingleSceneCollector] Initializing... COARSE={COARSE}, LIGHT={LIGHT}")
        self.aligner = GridAligner()
        self.sentinel_fetcher = SentinelFetcher(max_nodata_fraction=0.30 if COARSE else 0.20)
        self.index_calc = IndexCalculator()
        
        # Обмежуємо OSM лише Іспанією та Португалією, щоб він не ліз у Францію чи Марокко!
        self.spatial_fetcher = SpatialFeatureFetcher(osm_mode=osm_mode, allowed_countries=["ES", "PT"])
        
        self.target_builder = TargetBuilder(pixel_size_m=PIXEL_SCALE_METERS)
        if not LIGHT:
            self.population_fetcher = PopulationFetcher()
            self.era5_fetcher = ERA5Fetcher()
            self.weather_fetcher = WeatherFetcher()
            self.cffdrs_fetcher = CFFDRSFetcher()
        print("[SingleSceneCollector] Initialization complete.")

    def collect_sample(
        self,
        lat: float,
        lon: float,
        target_date: str,
        is_fire: int | bool = 1,
    ) -> Optional[Dict[str, Any]]:
        scene_id = f"SCENE_{lat:.4f}_{lon:.4f}_{target_date.replace('-', '')}"
        print(f"\n--- [COLLECT_SAMPLE] Starting collection for {scene_id} ---")
        total_start = time.perf_counter()
        timings: Dict[str, float] = {}

        grid_info = self.aligner.get_master_grid_info(lat, lon)
        target_year = datetime.strptime(target_date, "%Y-%m-%d").year

        pre_fire_dt = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)
        pre_fire_date = pre_fire_dt.strftime("%Y-%m-%d")

        print(f"[{scene_id}] 1. Fetching Sentinel data (radar & optical)...")
        t_s2 = time.perf_counter()
        sentinel_data = self.sentinel_fetcher.fetch_all_radar_optical(lat, lon, target_date)
        timings["sentinel_fetch"] = time.perf_counter() - t_s2
        print(f"[{scene_id}] Sentinel fetch finished in {timings['sentinel_fetch']:.2f}s")

        if not sentinel_data:
            print(f"[{scene_id}] -> FAILED: No Sentinel data returned.")
            return None

        bands_t0 = sentinel_data.get("bands_t0")
        t0_date = sentinel_data.get("t0_date")
        
        if not bands_t0 or not t0_date:
            print(f"[{scene_id}] -> FAILED: Missing T0 bands or T0 date.")
            return None
            
        print(f"[{scene_id}] Found valid Sentinel scene for T0: {t0_date}")

        masks_t0 = sentinel_data.get("masks_t0") or {}
        sar_bands = sentinel_data.get("sar_bands") or {}

        def _fetch_dem_osm():
            print(f"  [{scene_id}] [Thread] Starting DEM/OSM fetch...")
            t0 = time.perf_counter()
            res = self.spatial_fetcher.fetch_all_spatial_features(
                lat=lat,
                lon=lon,
                target_date=target_date,
                scl_10m=None,
            )
            elapsed = time.perf_counter() - t0
            print(f"  [{scene_id}] [Thread] DEM/OSM fetch done in {elapsed:.2f}s")
            timings["dem_osm_fetch"] = elapsed
            return res

        def _fetch_population_1d():
            print(f"  [{scene_id}] [Thread] Starting Population fetch...")
            t0 = time.perf_counter()
            res = self.population_fetcher.fetch_population_1d(
                lat=lat, lon=lon, year=min(target_year, 2020)
            )
            elapsed = time.perf_counter() - t0
            print(f"  [{scene_id}] [Thread] Population fetch done in {elapsed:.2f}s")
            timings["population_gee_1d"] = elapsed
            return res

        def _fetch_soil():
            print(f"  [{scene_id}] [Thread] Starting ERA5 Soil Moisture fetch...")
            t0 = time.perf_counter()
            res = self.era5_fetcher.fetch_soil_moisture(
                grid_info=grid_info, target_date=pre_fire_date
            )
            elapsed = time.perf_counter() - t0
            print(f"  [{scene_id}] [Thread] ERA5 Soil Moisture fetch done in {elapsed:.2f}s")
            timings["soil_moisture_gee"] = elapsed
            return res

        def _fetch_cffdrs():
            print(f"  [{scene_id}] [Thread] Starting CFFDRS fetch...")
            t0 = time.perf_counter()
            res = self.cffdrs_fetcher.fetch_cffdrs_metrics(
                lat=lat, lon=lon, date_t0=pre_fire_date
            )
            elapsed = time.perf_counter() - t0
            print(f"  [{scene_id}] [Thread] CFFDRS fetch done in {elapsed:.2f}s")
            timings["cffdrs_calc"] = elapsed
            return res

        def _fetch_weather_t0():
            print(f"  [{scene_id}] [Thread] Starting Weather T0 fetch...")
            t0 = time.perf_counter()
            res = self.weather_fetcher.fetch_target_day_metrics(
                lat=lat, lon=lon, target_date=pre_fire_date
            )
            elapsed = time.perf_counter() - t0
            print(f"  [{scene_id}] [Thread] Weather T0 fetch done in {elapsed:.2f}s")
            timings["weather_t0"] = elapsed
            return res

        print(f"[{scene_id}] 2. Executing concurrent fetchers (LIGHT={LIGHT})...")
        if LIGHT:
            spatial_features = _fetch_dem_osm()
            pop_context_1d = {}
            soil_moisture = np.full(grid_info["shape"], 0.25, dtype=np.float32)
            cffdrs_metrics = {}
            weather_t0 = {}
            tri_intervals = {}
        else:
            # ПОВЕРНУЛИ ПАРАЛЕЛЬНІ ПОТОКИ
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {
                    pool.submit(_fetch_dem_osm): "spatial",
                    pool.submit(_fetch_population_1d): "population",
                    pool.submit(_fetch_soil): "soil",
                    pool.submit(_fetch_cffdrs): "cffdrs",
                    pool.submit(_fetch_weather_t0): "weather_t0"
                }
                
                results = {}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                        print(f"  [{scene_id}] Thread '{name}' completed.")
                    except Exception as e:
                        print(f"  [{scene_id}] ERROR in thread '{name}': {e}")
                        results[name] = {} if name in ["spatial", "weather_t0", "cffdrs", "population"] else None

                spatial_features = results.get("spatial") or {}
                pop_context_1d = results.get("population") or {}
                soil_moisture = results.get("soil")
                if soil_moisture is None:
                    soil_moisture = np.full(grid_info["shape"], np.nan, dtype=np.float32)
                cffdrs_metrics = results.get("cffdrs") or {}
                weather_t0 = results.get("weather_t0") or {}

            print(f"[{scene_id}] 3. Fetching Tri-interval weather metrics...")
            t_tri = time.perf_counter()
            t0_w = (pre_fire_dt - timedelta(days=7)).strftime("%Y-%m-%d")
            tprev_w = (pre_fire_dt - timedelta(days=21)).strftime("%Y-%m-%d")
            tri_intervals = self.weather_fetcher.fetch_tri_interval_metrics(
                lat=lat, lon=lon, target_date=pre_fire_date, t0_date=t0_w, tprev_date=tprev_w
            )
            timings["weather_tri_intervals"] = time.perf_counter() - t_tri
            print(f"[{scene_id}] Tri-interval metrics done in {timings['weather_tri_intervals']:.2f}s")

        print(f"[{scene_id}] 4. Computing indices...")
        t_ind = time.perf_counter()
        indices = self.index_calc.compute_all_indices(
            b02_t0=bands_t0.get("B02"),
            b03_t0=bands_t0.get("B03"),
            b04_t0=bands_t0.get("B04"),
            b05_t0=bands_t0.get("B05"),
            b06_t0=bands_t0.get("B06"),
            b07_t0=bands_t0.get("B07"),
            b08_t0=bands_t0.get("B08"),
            b8a_t0=bands_t0.get("B8A"),
            b11_t0=bands_t0.get("B11"),
            b12_t0=bands_t0.get("B12"),
            scl_t0=None,
            sar_vv=sar_bands.get("SAR_VV"),
            sar_vh=sar_bands.get("SAR_VH"),
        )
        clean_indices = {
            k: v for k, v in indices.items() if isinstance(v, np.ndarray) and v.ndim == 2
        }
        timings["indices_calc"] = time.perf_counter() - t_ind
        print(f"[{scene_id}] Indices computed in {timings['indices_calc']:.2f}s")

        print(f"[{scene_id}] 5. Building target tensor and loss mask...")
        t_tgt = time.perf_counter()
        if LIGHT:
            target_tensor = np.zeros((4, *grid_info["shape"]), dtype=np.float32)
        else:
            target_tensor = self.target_builder.build_target(
                grid_info=grid_info, lat=lat, lon=lon, is_fire=is_fire
            )

        raw_invalid_t0 = masks_t0.get("MASK_INVALID", np.zeros(grid_info["shape"], dtype=np.float32))
        snow_mask = masks_t0.get("MASK_SNOW", np.zeros(grid_info["shape"], dtype=np.float32))
        loss_mask = np.where(snow_mask > 0.5, 0.0, raw_invalid_t0).astype(np.float32)

        timings["target_builder"] = time.perf_counter() - t_tgt
        print(f"[{scene_id}] Targets built in {timings['target_builder']:.2f}s")

        total_elapsed = time.perf_counter() - total_start
        timings["total_pipeline_time"] = total_elapsed

        print(f"[{scene_id}] Assembling final raster dictionary...")
        raw_rasters: Dict[str, Any] = {
            **bands_t0,
            **sar_bands,
            **clean_indices,
            **spatial_features,
            "Soil_Moisture": soil_moisture,
            "MASK_WATER": masks_t0.get("MASK_WATER", np.zeros(grid_info["shape"], dtype=np.uint8)),
            "MASK_SNOW": masks_t0.get("MASK_SNOW", np.zeros(grid_info["shape"], dtype=np.uint8)),
            "MASK_CLOUDS": masks_t0.get("MASK_CLOUDS", np.zeros(grid_info["shape"], dtype=np.uint8)),
            "MASK_CLOUD_SHADOWS": masks_t0.get("MASK_CLOUD_SHADOWS", np.zeros(grid_info["shape"], dtype=np.uint8)),
        }
        rasters_2d = {
            k: v for k, v in raw_rasters.items() if isinstance(v, np.ndarray) and v.ndim == 2
        }

        context_1d = {**weather_t0, **tri_intervals, **cffdrs_metrics, **pop_context_1d}
        if not context_1d:
            context_1d = {"light_dummy": 0.0}

        print(f"--- [COLLECT_SAMPLE] Completed {scene_id} in {total_elapsed:.2f}s ---")
        return {
            "metadata": {
                "lat": lat,
                "lon": lon,
                "target_date": target_date,
                "pre_fire_date": pre_fire_date,
                "t0_date": t0_date,
                "is_fire": int(is_fire),
                "crs": str(grid_info["crs"]),
                "utm_bounds": grid_info["utm_bounds"],
                "elapsed_seconds": total_elapsed,
                "timings": timings,
            },
            "rasters_2d": rasters_2d,
            "context_1d": context_1d,
            "loss_mask": loss_mask,
            "scl": bands_t0["SCL"].astype(np.uint8),
            "target": target_tensor,
        }