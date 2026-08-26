from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time
from typing import Any, Dict, Optional
import numpy as np

from src.data_pipeline.cffdrs_fetcher import CFFDRSFetcher
from src.data_pipeline.era5_fetcher import ERA5Fetcher
from src.data_pipeline.index_calculator import IndexCalculator
from src.data_pipeline.nightlight_fetcher import NightlightFetcher
from src.data_pipeline.osm_dem_fetcher import SpatialFeatureFetcher
from src.data_pipeline.population_fetcher import PopulationFetcher
from src.data_pipeline.sentinel_fetcher import SentinelFetcher
from src.data_pipeline.target_builder import TargetBuilder
from src.data_pipeline.weather_fetcher import WeatherFetcher
from src.processing.grid_aligner import GridAligner


class SingleSceneCollector:
    """orchestrates multi-modal geospatial and multi-interval meteorological data collection."""

    def __init__(self):
        self.aligner = GridAligner()
        self.sentinel_fetcher = SentinelFetcher()
        self.index_calc = IndexCalculator()
        self.spatial_fetcher = SpatialFeatureFetcher()
        self.nightlight_fetcher = NightlightFetcher()
        self.population_fetcher = PopulationFetcher()
        self.era5_fetcher = ERA5Fetcher()
        self.weather_fetcher = WeatherFetcher()
        self.cffdrs_fetcher = CFFDRSFetcher()
        self.target_builder = TargetBuilder(pixel_size_m=10.0)

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
        target_year = datetime.strptime(target_date, "%Y-%m-%d").year

        def _fetch_sentinel():
            t0 = time.perf_counter()
            res = self.sentinel_fetcher.fetch_all_radar_optical(lat, lon, target_date)
            timings["sentinel_fetch"] = time.perf_counter() - t0
            return res

        def _fetch_dem_osm():
            t0 = time.perf_counter()
            res = self.spatial_fetcher.fetch_all_spatial_features(lat, lon, scl_10m=None)
            timings["dem_osm_fetch"] = time.perf_counter() - t0
            return res

        def _fetch_nightlight():
            t0 = time.perf_counter()
            res = self.nightlight_fetcher.fetch_nightlight_potential(grid_info=grid_info, year=target_year)
            timings["nightlight_gee"] = time.perf_counter() - t0
            return res

        def _fetch_population():
            t0 = time.perf_counter()
            res = self.population_fetcher.fetch_population(grid_info=grid_info, year=min(target_year, 2020))
            timings["population_gee"] = time.perf_counter() - t0
            return res

        def _fetch_soil():
            t0 = time.perf_counter()
            res = self.era5_fetcher.fetch_soil_moisture(grid_info=grid_info, target_date=target_date)
            timings["soil_moisture_gee"] = time.perf_counter() - t0
            return res

        def _fetch_cffdrs():
            t0 = time.perf_counter()
            res = self.cffdrs_fetcher.fetch_cffdrs_metrics(lat=lat, lon=lon, date_t0=target_date)
            timings["cffdrs_calc"] = time.perf_counter() - t0
            return res

        def _fetch_weather_t0():
            t0 = time.perf_counter()
            res = self.weather_fetcher.fetch_target_day_metrics(lat=lat, lon=lon, target_date=target_date)
            timings["weather_t0"] = time.perf_counter() - t0
            return res

        with ThreadPoolExecutor(max_workers=7) as pool:
            f_sentinel = pool.submit(_fetch_sentinel)
            f_spatial = pool.submit(_fetch_dem_osm)
            f_nightlight = pool.submit(_fetch_nightlight)
            f_population = pool.submit(_fetch_population)
            f_soil = pool.submit(_fetch_soil)
            f_cffdrs = pool.submit(_fetch_cffdrs)
            f_weather = pool.submit(_fetch_weather_t0)

            sentinel_data = f_sentinel.result()
            spatial_features = f_spatial.result()
            nightlight_potential = f_nightlight.result()
            pop_potential = f_population.result()
            soil_moisture = f_soil.result()
            cffdrs_metrics = f_cffdrs.result()
            weather_t0 = f_weather.result()

        if not sentinel_data:
            return None

        t0_date = sentinel_data["t0_date"]
        tprev_date = sentinel_data["tprev_date"]
        bands_t0 = sentinel_data["bands_t0"]
        masks_t0 = sentinel_data["masks_t0"]
        bands_tprev = sentinel_data["bands_tprev"]
        sar_bands = sentinel_data["sar_bands"]

        # 3-interval meteorological metrics
        t_tri = time.perf_counter()
        tri_intervals = self.weather_fetcher.fetch_tri_interval_metrics(
            lat=lat, lon=lon, target_date=target_date, t0_date=t0_date, tprev_date=tprev_date
        )
        timings["weather_tri_intervals"] = time.perf_counter() - t_tri

        # compute spectral and radar indices
        t_ind = time.perf_counter()
        indices = self.index_calc.compute_all_indices(
            b02_t0=bands_t0.get("B02"),
            b04_t0=bands_t0.get("B04"),
            b08_t0=bands_t0.get("B08"),
            b8a_t0=bands_t0.get("B8A"),
            b11_t0=bands_t0.get("B11"),
            b12_t0=bands_t0.get("B12"),
            scl_t0=None,
            b05_t0=bands_t0.get("B05"),
            b04_tprev=bands_tprev.get("B04"),
            b08_tprev=bands_tprev.get("B08"),
            b11_tprev=bands_tprev.get("B11"),
            b12_tprev=bands_tprev.get("B12"),
            sar_vv=sar_bands.get("SAR_VV"),
            sar_vh=sar_bands.get("SAR_VH"),
        )
        clean_indices = {k: v for k, v in indices.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        timings["indices_calc"] = time.perf_counter() - t_ind

        # target tensor generation
        t_tgt = time.perf_counter()
        target_tensor = self.target_builder.build_target(grid_info=grid_info, lat=lat, lon=lon, is_fire=is_fire)
        loss_mask = masks_t0["MASK_INVALID"]
        timings["target_builder"] = time.perf_counter() - t_tgt

        total_elapsed = time.perf_counter() - total_start
        timings["total_pipeline_time"] = total_elapsed

        raw_rasters: Dict[str, Any] = {
            **bands_t0,
            **sar_bands,
            **clean_indices,
            **spatial_features,
            "Nightlight_Potential": nightlight_potential,
            "Population_Potential": pop_potential,
            "Soil_Moisture": soil_moisture,
            "MASK_WATER": masks_t0["MASK_WATER"],
            "MASK_SNOW": masks_t0["MASK_SNOW"],
            "MASK_CLOUDS": masks_t0["MASK_CLOUDS"],
            "MASK_CLOUD_SHADOWS": masks_t0["MASK_CLOUD_SHADOWS"],
        }
        rasters_2d = {k: v for k, v in raw_rasters.items() if isinstance(v, np.ndarray) and v.ndim == 2}

        # combine all 1d features into one clean dictionary
        context_1d = {
            **weather_t0,
            **tri_intervals,
            **cffdrs_metrics,
        }

        return {
            "metadata": {
                "lat": lat,
                "lon": lon,
                "target_date": target_date,
                "t0_date": t0_date,
                "tprev_date": tprev_date,
                "is_fire": int(is_fire),
                "crs": str(grid_info["crs"]),
                "utm_bounds": grid_info["utm_bounds"],
                "elapsed_seconds": total_elapsed,
                "timings": timings,
            },
            "rasters_2d": rasters_2d,
            "context_1d": context_1d,
            "loss_mask": loss_mask,
            "target": target_tensor,
        }