from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
    """Orchestrates multi-modal geospatial and multi-interval meteorological data collection."""

    def __init__(self, osm_mode: str = "local_history"):
        self.aligner = GridAligner()
        self.sentinel_fetcher = SentinelFetcher()
        self.index_calc = IndexCalculator()
        self.spatial_fetcher = SpatialFeatureFetcher(osm_mode=osm_mode)
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

        t_s2 = time.perf_counter()
        sentinel_data = self.sentinel_fetcher.fetch_all_radar_optical(lat, lon, target_date)
        timings["sentinel_fetch"] = time.perf_counter() - t_s2

        if not sentinel_data:
            return None

        bands_t0 = sentinel_data.get("bands_t0")
        bands_tprev = sentinel_data.get("bands_tprev")
        t0_date = sentinel_data.get("t0_date")
        tprev_date = sentinel_data.get("tprev_date")

        if not bands_t0 or not bands_tprev or not t0_date or not tprev_date:
            return None

        masks_t0 = sentinel_data.get("masks_t0") or {}
        sar_bands = sentinel_data.get("sar_bands") or {}

        def _fetch_dem_osm():
            t0 = time.perf_counter()
            res = self.spatial_fetcher.fetch_all_spatial_features(
                lat=lat,
                lon=lon,
                target_date=target_date,
                scl_10m=None,
            )
            timings["dem_osm_fetch"] = time.perf_counter() - t0
            return res

        def _fetch_nightlight():
            t0 = time.perf_counter()
            res = self.nightlight_fetcher.fetch_nightlight_potential(
                grid_info=grid_info, year=target_year
            )
            timings["nightlight_gee"] = time.perf_counter() - t0
            return res

        def _fetch_population():
            t0 = time.perf_counter()
            res = self.population_fetcher.fetch_population(
                grid_info=grid_info, year=min(target_year, 2020)
            )
            timings["population_gee"] = time.perf_counter() - t0
            return res

        def _fetch_soil():
            t0 = time.perf_counter()
            res = self.era5_fetcher.fetch_soil_moisture(
                grid_info=grid_info, target_date=target_date
            )
            timings["soil_moisture_gee"] = time.perf_counter() - t0
            return res

        def _fetch_cffdrs():
            t0 = time.perf_counter()
            res = self.cffdrs_fetcher.fetch_cffdrs_metrics(
                lat=lat, lon=lon, date_t0=target_date
            )
            timings["cffdrs_calc"] = time.perf_counter() - t0
            return res

        def _fetch_weather_t0():
            t0 = time.perf_counter()
            res = self.weather_fetcher.fetch_target_day_metrics(
                lat=lat, lon=lon, target_date=target_date
            )
            timings["weather_t0"] = time.perf_counter() - t0
            return res

        with ThreadPoolExecutor(max_workers=6) as pool:
            f_spatial = pool.submit(_fetch_dem_osm)
            f_nightlight = pool.submit(_fetch_nightlight)
            f_population = pool.submit(_fetch_population)
            f_soil = pool.submit(_fetch_soil)
            f_cffdrs = pool.submit(_fetch_cffdrs)
            f_weather = pool.submit(_fetch_weather_t0)

            spatial_features = f_spatial.result()
            nightlight_potential = f_nightlight.result()
            pop_potential = f_population.result()
            soil_moisture = f_soil.result()
            cffdrs_metrics = f_cffdrs.result()
            weather_t0 = f_weather.result()

        t_tri = time.perf_counter()
        tri_intervals = self.weather_fetcher.fetch_tri_interval_metrics(
            lat=lat, lon=lon, target_date=target_date, t0_date=t0_date, tprev_date=tprev_date
        )
        timings["weather_tri_intervals"] = time.perf_counter() - t_tri

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
            b04_tprev=bands_tprev.get("B04"),
            b08_tprev=bands_tprev.get("B08"),
            b11_tprev=bands_tprev.get("B11"),
            b12_tprev=bands_tprev.get("B12"),
            sar_vv=sar_bands.get("SAR_VV"),
            sar_vh=sar_bands.get("SAR_VH"),
        )
        clean_indices = {
            k: v for k, v in indices.items() if isinstance(v, np.ndarray) and v.ndim == 2
        }
        timings["indices_calc"] = time.perf_counter() - t_ind

        t_tgt = time.perf_counter()
        target_tensor = self.target_builder.build_target(
            grid_info=grid_info, lat=lat, lon=lon, is_fire=is_fire
        )
        raw_invalid = masks_t0.get(
            "MASK_INVALID", np.zeros(grid_info["shape"], dtype=np.float32)
        )
        snow_mask = masks_t0.get("MASK_SNOW", np.zeros(grid_info["shape"], dtype=np.float32))
        loss_mask = np.where(snow_mask > 0.5, 0.0, raw_invalid).astype(np.float32)
        
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
            "MASK_WATER": masks_t0.get("MASK_WATER", np.zeros(grid_info["shape"], dtype=np.uint8)),
            "MASK_SNOW": masks_t0.get("MASK_SNOW", np.zeros(grid_info["shape"], dtype=np.uint8)),
            "MASK_CLOUDS": masks_t0.get("MASK_CLOUDS", np.zeros(grid_info["shape"], dtype=np.uint8)),
            "MASK_CLOUD_SHADOWS": masks_t0.get("MASK_CLOUD_SHADOWS", np.zeros(grid_info["shape"], dtype=np.uint8)),
        }
        rasters_2d = {
            k: v for k, v in raw_rasters.items() if isinstance(v, np.ndarray) and v.ndim == 2
        }

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
    
if __name__ == "__main__":
    from src.data_pipeline.patch_extractor import PatchExtractor

    # initialize collector and patch extractor
    collector = SingleSceneCollector()
    extractor = PatchExtractor(patch_size=256, stride=256, max_invalid_ratio=0.20)

    # test wildfire scene with both optical and sar coverage (jasper, alberta)
    test_lat = 52.5708739
    test_lon = -117.9518471
    test_date = "2021-08-15"

    print(f"=== fetching scene [{test_lat}, {test_lon}] on {test_date} ===")
    sample = collector.collect_sample(
        lat=test_lat, lon=test_lon, target_date=test_date, is_fire=1
    )

    if not sample:
        print("[error] failed to collect scene or missing sentinel-2 optical pair.")
        exit(1)

    # print 1d context feature vector to console
    context_1d = sample["context_1d"]
    print("\n" + "=" * 65)
    print(f" 1D CONTEXT VECTOR (total features: {len(context_1d)})")
    print("=" * 65)
    for key in sorted(context_1d.keys()):
        val = context_1d[key]
        if isinstance(val, (int, float, np.floating)):
            print(f"  {key:<32}: {val:>12.4f}")
        else:
            print(f"  {key:<32}: {str(val):>12}")
    print("=" * 65 + "\n")

    # extract patches using patchextractor
    scene_id = f"SCENE_{test_lat:.4f}_{test_lon:.4f}_{test_date.replace('-', '')}"
    patches = list(extractor.extract_patches(sample, scene_id=scene_id))
    print(f"extracted {len(patches)} valid patches out of 64 possible (max invalid ratio <= 20%).")

    if not patches:
        print("[warning] zero patches passed quality threshold.")
        exit(0)

    # sequential visualization of 2d channels on reconstructed master grid
    channel_names = patches[0]["channel_names_2d"]
    num_channels = len(channel_names)
    full_h, full_w = sample["loss_mask"].shape

    print(f"\nstarting visualization for {num_channels} channels.")
    print("close current window to display next channel...\n")

    for ch_idx, ch_name in enumerate(channel_names):
        # empty canvas initialized with nan so rejected patches stay transparent/blank
        canvas = np.full((full_h, full_w), np.nan, dtype=np.float32)

        # place each valid 256x256 patch into its original grid position
        for p in patches:
            r = p["metadata"]["row_offset"]
            c = p["metadata"]["col_offset"]
            canvas[r : r + 256, c : c + 256] = p["X_2d"][ch_idx]

        fig, ax = plt.subplots(figsize=(9, 9))

        # select colormap based on sensor and feature type
        if "MASK" in ch_name or "SCL" in ch_name:
            cmap = "gray"
        elif "SAR" in ch_name:
            cmap = "plasma"
        elif any(idx in ch_name for idx in ["NDVI", "EVI", "NDRE"]):
            cmap = "YlGn"
        elif any(idx in ch_name for idx in ["NDMI", "MSI", "NMDI", "NBR"]):
            cmap = "coolwarm"
        elif ch_name in ["Elevation", "Slope"]:
            cmap = "terrain"
        else:
            cmap = "viridis"

        # compute robust display percentiles excluding nan values
        valid_pixels = canvas[np.isfinite(canvas)]
        if len(valid_pixels) > 0 and "MASK" not in ch_name:
            vmin, vmax = np.percentile(valid_pixels, [2, 98])
        else:
            vmin = np.nanmin(canvas) if len(valid_pixels) > 0 else 0.0
            vmax = np.nanmax(canvas) if len(valid_pixels) > 0 else 1.0

        im = ax.imshow(canvas, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)

        # draw 8x8 patch grid lines (every 256 pixels)
        for offset in range(0, full_h + 1, 256):
            ax.axhline(offset - 0.5, color="red", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.axvline(offset - 0.5, color="red", linestyle="--", linewidth=0.7, alpha=0.7)

        ax.set_title(
            f"[{ch_idx + 1}/{num_channels}] channel: {ch_name}\n"
            f"valid patches: {len(patches)}/64 | canvas shape: {full_h}x{full_w}",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("x pixel (10m)")
        ax.set_ylabel("y pixel (10m)")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("value")

        plt.tight_layout()
        plt.show()  # blocks execution until window is closed
        plt.close(fig)

    print("visualization completed.")