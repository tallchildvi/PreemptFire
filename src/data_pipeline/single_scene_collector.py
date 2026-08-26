from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
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
    """Fully parallelized scene data orchestrator with real-time thread logging."""

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

        print("\n" + "=" * 75)
        print(f" PARALLEL PIPELINE START: ({lat:.4f}, {lon:.4f}) | Target Date: {target_date}")
        print("=" * 75)

        grid_info = self.aligner.get_master_grid_info(lat, lon)
        target_year = datetime.strptime(target_date, "%Y-%m-%d").year

        def _fetch_sentinel():
            t0 = time.perf_counter()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-Sentinel] Fetching Sentinel-2 & SAR...")
            res = self.sentinel_fetcher.fetch_all_radar_optical(lat, lon, target_date)
            dur = time.perf_counter() - t0
            timings["sentinel_fetch"] = dur
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-Sentinel] Finished in {dur:.2f}s")
            return res

        def _fetch_dem_osm():
            t0 = time.perf_counter()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-DEM/OSM] Querying OSM & DEM features...")
            res = self.spatial_fetcher.fetch_all_spatial_features(lat, lon, scl_10m=None)
            dur = time.perf_counter() - t0
            timings["dem_osm_fetch"] = dur
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-DEM/OSM] Finished in {dur:.2f}s")
            return res

        def _fetch_nightlight():
            t0 = time.perf_counter()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-VIIRS] Fetching nightlight...")
            res = self.nightlight_fetcher.fetch_nightlight_potential(grid_info=grid_info, year=target_year)
            dur = time.perf_counter() - t0
            timings["nightlight_gee"] = dur
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-VIIRS] Finished in {dur:.2f}s")
            return res

        def _fetch_population():
            t0 = time.perf_counter()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-WorldPop] Fetching population...")
            res = self.population_fetcher.fetch_population(grid_info=grid_info, year=min(target_year, 2020))
            dur = time.perf_counter() - t0
            timings["population_gee"] = dur
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-WorldPop] Finished in {dur:.2f}s")
            return res

        def _fetch_soil():
            t0 = time.perf_counter()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-ERA5] Fetching soil moisture...")
            res = self.era5_fetcher.fetch_soil_moisture(grid_info=grid_info, target_date=target_date)
            dur = time.perf_counter() - t0
            timings["soil_moisture_gee"] = dur
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-ERA5] Finished in {dur:.2f}s")
            return res

        def _fetch_cffdrs():
            t0 = time.perf_counter()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-CFFDRS] Computing 90-day spinup...")
            res = self.cffdrs_fetcher.fetch_cffdrs_metrics(lat=lat, lon=lon, date_t0=target_date)
            dur = time.perf_counter() - t0
            timings["cffdrs_calc"] = dur
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-CFFDRS] Finished in {dur:.2f}s")
            return res

        def _fetch_weather():
            t0 = time.perf_counter()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-Weather] Fetching Open-Meteo...")
            res = self.weather_fetcher.fetch_target_day_metrics(lat=lat, lon=lon, target_date=target_date)
            dur = time.perf_counter() - t0
            timings["weather_t0"] = dur
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] [Thread-Weather] Finished in {dur:.2f}s")
            return res

        # Run all data providers simultaneously from second 0
        with ThreadPoolExecutor(max_workers=7) as pool:
            f_sentinel = pool.submit(_fetch_sentinel)
            f_spatial = pool.submit(_fetch_dem_osm)
            f_nightlight = pool.submit(_fetch_nightlight)
            f_population = pool.submit(_fetch_population)
            f_soil = pool.submit(_fetch_soil)
            f_cffdrs = pool.submit(_fetch_cffdrs)
            f_weather = pool.submit(_fetch_weather)

            sentinel_data = f_sentinel.result()
            spatial_features = f_spatial.result()
            nightlight_potential = f_nightlight.result()
            pop_potential = f_population.result()
            soil_moisture = f_soil.result()
            cffdrs_metrics = f_cffdrs.result()
            weather_t0 = f_weather.result()

        if not sentinel_data:
            print("  [Error] Failed to fetch Sentinel data. Aborting.")
            return None

        t0_date = sentinel_data["t0_date"]
        tprev_date = sentinel_data["tprev_date"]
        bands_t0 = sentinel_data["bands_t0"]
        masks_t0 = sentinel_data["masks_t0"]
        bands_tprev = sentinel_data["bands_tprev"]
        sar_bands = sentinel_data["sar_bands"]

        # Fetch interval weather metrics between Tprev and T0
        t_int = time.perf_counter()
        weather_interval = self.weather_fetcher.fetch_interval_metrics(
            lat=lat, lon=lon, start_date=tprev_date, end_date=t0_date
        )
        timings["weather_interval"] = time.perf_counter() - t_int

        # Spectral and temporal vegetation index calculation
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

        # Continuous target field generation
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

        context_1d = {
            **weather_interval,
            **weather_t0,
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
                "crs": grid_info["crs"],
                "utm_bounds": grid_info["utm_bounds"],
                "elapsed_seconds": total_elapsed,
                "timings": timings,
            },
            "rasters_2d": rasters_2d,
            "context_1d": context_1d,
            "loss_mask": loss_mask,
            "target": target_tensor,
        }


def print_1d_and_performance_metrics(sample: Dict[str, Any]):
    meta = sample["metadata"]
    c1d = sample["context_1d"]
    timings = meta["timings"]

    print("\n" + "=" * 70)
    print(" EXECUTION PERFORMANCE BREAKDOWN")
    print("=" * 70)
    for stage, sec in timings.items():
        if stage == "total_pipeline_time":
            continue
        print(f"  {stage:<30}: {sec:6.2f}s")
    print("-" * 70)
    print(f"  TOTAL EXTRACTION TIME (WALL-CLOCK): {meta['elapsed_seconds']:6.2f}s ({meta['elapsed_seconds']/60:.2f} min)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(" SCENE METADATA & 1D CONTEXT VECTORS")
    print("=" * 70)
    print(f"  Target Date          : {meta['target_date']}")
    print(f"  T0 Satellite Date    : {meta['t0_date']} (Tprev: {meta['tprev_date']})")
    print(f"  Coordinates          : [{meta['lat']:.4f}, {meta['lon']:.4f}] ({meta['crs']})")
    print(f"  Fire Ground Truth    : {'POSITIVE (Fire Event)' if meta['is_fire'] else 'NEGATIVE (No Fire)'}")
    print("-" * 70)
    print("  1D TABULAR CONTEXT FEATURES:")
    print("-" * 70)
    for key, val in sorted(c1d.items()):
        print(f"    {key:<28}: {val:8.3f}")
    print("=" * 65 + "\n")


def visualize_2d_layers_step_by_step(sample: Dict[str, Any], downsample_step: int = 2):
    rasters = sample["rasters_2d"]
    loss_mask = sample["loss_mask"]
    target = sample["target"]

    layers_to_plot: List[Tuple[str, np.ndarray, str]] = []

    for name, arr in rasters.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            continue
        if "MASK" in name:
            cmap = "gray"
        elif "SAR" in name:
            cmap = "plasma"
        elif any(k in name for k in ["NDVI", "EVI", "NDRE"]):
            cmap = "RdYlGn"
        elif any(k in name for k in ["NDMI", "Soil_Moisture", "Water"]):
            cmap = "Blues"
        elif any(k in name for k in ["Slope", "Elevation", "Travel_Time", "Dist_"]):
            cmap = "terrain"
        else:
            cmap = "viridis"
        layers_to_plot.append((f"Feature: {name}", arr, cmap))

    if isinstance(loss_mask, np.ndarray) and loss_mask.ndim == 2:
        layers_to_plot.append(("Loss Mask (MASK_INVALID)", loss_mask, "binary_r"))

    if isinstance(target, np.ndarray) and target.ndim == 3:
        target_cmaps = ["inferno", "magma", "plasma", "viridis"]
        target_names = [
            "Target Ch 0 (σ=250m Local)",
            "Target Ch 1 (σ=1000m Sub-Regional)",
            "Target Ch 2 (σ=3000m Meso-Corridor)",
            "Target Ch 3 (σ=4000m Macro-Field)",
        ]
        for ch_idx in range(min(target.shape[0], len(target_names))):
            layers_to_plot.append((target_names[ch_idx], target[ch_idx], target_cmaps[ch_idx]))

    total_layers = len(layers_to_plot)
    total_steps = math.ceil(total_layers / 2)

    print(f"Displaying {total_layers} valid 2D layers across {total_steps} sequential windows.")

    for step_idx in range(total_steps):
        i1 = step_idx * 2
        i2 = i1 + 1

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        title1, arr1, cmap1 = layers_to_plot[i1]
        ds_arr1 = arr1[::downsample_step, ::downsample_step]
        im1 = axes[0].imshow(ds_arr1, cmap=cmap1, origin="upper")
        axes[0].set_title(
            f"[{i1 + 1}/{total_layers}] {title1}\n"
            f"Range: [{np.nanmin(arr1):.3f}, {np.nanmax(arr1):.3f}] | Mean: {np.nanmean(arr1):.3f}",
            fontsize=10,
            fontweight="bold",
        )
        axes[0].axis("off")
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=8)

        if i2 < total_layers:
            title2, arr2, cmap2 = layers_to_plot[i2]
            ds_arr2 = arr2[::downsample_step, ::downsample_step]
            im2 = axes[1].imshow(ds_arr2, cmap=cmap2, origin="upper")
            axes[1].set_title(
                f"[{i2 + 1}/{total_layers}] {title2}\n"
                f"Range: [{np.nanmin(arr2):.3f}, {np.nanmax(arr2):.3f}] | Mean: {np.nanmean(arr2):.3f}",
                fontsize=10,
                fontweight="bold",
            )
            axes[1].axis("off")
            cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=8)
        else:
            axes[1].axis("off")

        fig.suptitle(f"Step {step_idx + 1} of {total_steps}", fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    collector = SingleSceneCollector()

    test_lat, test_lon = 56.7264, -111.3803
    test_date = "2025-06-15"

    sample = collector.collect_sample(
        lat=test_lat,
        lon=test_lon,
        target_date=test_date,
        is_fire=1,
    )

    if sample:
        print_1d_and_performance_metrics(sample)
        visualize_2d_layers_step_by_step(sample, downsample_step=2)