import os
from typing import Any, Dict, Optional, Tuple

import ee
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject
from scipy.ndimage import white_tophat
from scipy.signal import fftconvolve

from src.processing.grid_aligner import GridAligner


class NightlightFetcher:
    """
    Fetches stray-light-corrected VIIRS DNB monthly composite radiance from GEE
    and generates an isolated anthropogenic activity potential field.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        buffer_meters: float = 30_000.0,
        decay_distance_meters: float = 4_000.0,
        alpha: float = 1.0,
        floor_value: float = 1e-4,
        reference_radiance: float = 500.0,
        aurora_max_cv: float = 0.25,
        aurora_min_peak_nw: float = 1.5,
        aurora_min_hot_fraction: float = 0.001,
        tophat_radius_meters: float = 2_500.0,
    ):
        load_dotenv()
        self.aligner = GridAligner()
        self.project_id = project_id or os.getenv("GEE_PROJECT")

        self.buffer_meters = buffer_meters
        self.decay_distance_meters = decay_distance_meters
        self.alpha = alpha
        self.floor_value = floor_value
        self.reference_radiance = reference_radiance

        # Aurora rejection parameters
        self.aurora_max_cv = aurora_max_cv
        self.aurora_min_peak_nw = aurora_min_peak_nw
        self.aurora_min_hot_fraction = aurora_min_hot_fraction
        self.tophat_radius_meters = tophat_radius_meters

        self._init_earth_engine()
        self.collection_name = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"

    def _init_earth_engine(self) -> None:
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Earth Engine: {e}\n"
                f"Run 'ee.Authenticate()' or verify GEE_PROJECT in your .env"
            )

    def _get_composite_image(self, year: int) -> ee.Image:
        col = ee.ImageCollection(self.collection_name)

        # 1. Dark, snow-free autumn window (Sep-Oct) avoids both summer twilight and snow albedo
        autumn_filtered = (
            col.filter(ee.Filter.calendarRange(year, year, "year"))
            .filter(ee.Filter.calendarRange(8, 10, "month"))
            .select(["avg_rad"])
        )
        if autumn_filtered.size().getInfo() > 0:
            return autumn_filtered.median()

        # 2. Entire target year median
        year_filtered = (
            col.filter(ee.Filter.calendarRange(year, year, "year"))
            .select(["avg_rad"])
        )
        if year_filtered.size().getInfo() > 0:
            return year_filtered.median()

        # 3. Fallback to latest 12 available months
        return col.select(["avg_rad"]).sort("system:time_start", False).limit(12).median()

    def _white_tophat_filter(self, arr: np.ndarray, pixel_size_m: float) -> np.ndarray:
        r_px = max(1, int(round(self.tophat_radius_meters / pixel_size_m)))
        y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
        disk_struct = ((x**2 + y**2) <= r_px**2).astype(np.uint8)
        peaks = white_tophat(arr, structure=disk_struct)
        return peaks.astype(np.float32)

    def _evaluate_aurora_contamination(self, raw: np.ndarray, tophat_peaks: np.ndarray) -> Dict[str, Any]:
        mean_val = float(np.nanmean(raw)) + 1e-8
        std_val = float(np.nanstd(raw))
        cv = std_val / mean_val

        hot_pixel_fraction = float((tophat_peaks >= self.aurora_min_peak_nw).mean())
        peak_radiance = float(np.nanmax(tophat_peaks))

        # Scene is flagged as aurora if it is spatially uniform or lacks localized point peaks
        is_aurora = (
            cv < self.aurora_max_cv
            or hot_pixel_fraction < self.aurora_min_hot_fraction
            or peak_radiance < self.aurora_min_peak_nw
        )

        return {
            "cv": cv,
            "hot_fraction": hot_pixel_fraction,
            "peak_radiance": peak_radiance,
            "is_aurora": is_aurora,
        }

    def _make_lorentzian_kernel(self, pixel_size_m: float, max_radius_px: int) -> Tuple[np.ndarray, int]:
        # d_half is the distance where kernel value drops to 0.5
        d_half = self.decay_distance_meters
        beta = 1.0 / (d_half**2)

        r_px = min(int(np.ceil(6.0 * d_half / pixel_size_m)), max_radius_px)
        y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
        dist_sq = (x**2 + y**2) * (pixel_size_m**2)

        kernel = (1.0 / (1.0 + beta * dist_sq)).astype(np.float32)
        return kernel, r_px

    def fetch_raw_nightlight(self, grid_info: dict, year: int = 2023) -> np.ndarray:
        shape = grid_info["shape"]
        master_crs = grid_info["crs"]
        master_transform = grid_info["transform"]
        min_lon, min_lat, max_lon, max_lat = grid_info["bbox_wgs84"]

        buf = 0.01
        region = ee.Geometry.Rectangle(
            [min_lon - buf, min_lat - buf, max_lon + buf, max_lat + buf],
            proj="EPSG:4326",
            geodesic=False,
        )

        nl_image = self._get_composite_image(year=year)
        try:
            download_url = nl_image.getDownloadURL({
                "region": region,
                "scale": 500,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            })
            resp = requests.get(download_url, timeout=40)
            resp.raise_for_status()
        except Exception as e:
            print(f"[NightlightFetcher] Raw download failed: {e}")
            return np.zeros(shape, dtype=np.float32)

        raw_10m = np.zeros(shape, dtype=np.float32)
        with MemoryFile(resp.content) as memfile:
            with memfile.open() as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=raw_10m,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=master_crs,
                    dst_transform=master_transform,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=0.0,
                )

        raw_10m = np.nan_to_num(raw_10m, nan=0.0)
        return np.clip(raw_10m, 0.0, None).astype(np.float32)

    def fetch_nightlight_potential(
        self, grid_info: dict, year: int = 2023, return_none_on_aurora: bool = False
    ) -> Optional[np.ndarray]:
        shape = grid_info["shape"]
        master_crs = grid_info["crs"]
        master_transform = grid_info["transform"]
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]

        b = self.buffer_meters
        transformer = Transformer.from_crs(master_crs, "EPSG:4326", always_xy=True)
        corners_utm = [
            (min_x - b, min_y - b),
            (min_x - b, max_y + b),
            (max_x + b, min_y - b),
            (max_x + b, max_y + b),
        ]
        corners_wgs84 = [transformer.transform(x, y) for x, y in corners_utm]
        lons = [pt[0] for pt in corners_wgs84]
        lats = [pt[1] for pt in corners_wgs84]

        region = ee.Geometry.Rectangle(
            [min(lons), min(lats), max(lons), max(lats)],
            proj="EPSG:4326",
            geodesic=False,
        )

        nl_image = self._get_composite_image(year=year)
        try:
            download_url = nl_image.getDownloadURL({
                "region": region,
                "scale": 500,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            })
            resp = requests.get(download_url, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"[NightlightFetcher] GEE download failed: {e}")
            return None if return_none_on_aurora else np.full(shape, self.floor_value, dtype=np.float32)

        potential_10m = np.zeros(shape, dtype=np.float32)

        with MemoryFile(resp.content) as memfile:
            with memfile.open() as src:
                raw_500m = src.read(1).astype(np.float32)
                raw_500m = np.nan_to_num(raw_500m, nan=0.0)
                raw_500m = np.clip(raw_500m, 0.0, None)

                pixel_size_m = 500.0

                # 1. Morphological extraction of localized bright sources
                tophat_peaks = self._white_tophat_filter(raw_500m, pixel_size_m)

                # 2. Aurora & airglow screening
                diag = self._evaluate_aurora_contamination(raw_500m, tophat_peaks)
                if diag["is_aurora"]:
                    if return_none_on_aurora:
                        return None
                    return np.full(shape, self.floor_value, dtype=np.float32)

                # 3. Clean source selection
                sources = np.clip(tophat_peaks, 0.0, None)

                # 4. Lorentzian spatial convolution
                kernel, k_radius = self._make_lorentzian_kernel(
                    pixel_size_m=pixel_size_m,
                    max_radius_px=max(sources.shape) * 2,
                )

                padded = np.pad(sources, k_radius, mode="constant", constant_values=0.0)
                conv = fftconvolve(padded, kernel, mode="same").astype(np.float32)
                conv = conv[k_radius:-k_radius or None, k_radius:-k_radius or None]
                conv = np.clip(conv, 0.0, None)

                # 5. Michaelis-Menten bounded saturation
                ref_m = float(self.reference_radiance)
                potential_500m = (self.alpha * conv / (ref_m + conv)).astype(np.float32)

                # 6. Reproject to master 10m grid
                reproject(
                    source=potential_500m,
                    destination=potential_10m,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=master_crs,
                    dst_transform=master_transform,
                    resampling=Resampling.bilinear,
                    dst_nodata=0.0,
                )

        potential_10m = np.nan_to_num(potential_10m, nan=0.0, posinf=0.0, neginf=0.0)
        potential_10m = np.clip(potential_10m, 0.0, self.alpha)
        potential_10m = np.maximum(potential_10m, self.floor_value)

        return potential_10m.astype(np.float32)


if __name__ == "__main__":
    aligner = GridAligner()

    BENCHMARK_SCENES = {
        "Arctic Wilderness": (69.9286, -127.4445),
        "Edmonton Downtown": (53.5461, -113.4937),
        "Fort McMurray": (56.7264, -111.3803),
        "Remote Boreal": (59.3862, -108.8932),
    }

    year = 2023
    fetcher = NightlightFetcher(
        buffer_meters=30_000.0,
        decay_distance_meters=4_000.0,
        alpha=1.0,
        floor_value=1e-4,
        reference_radiance=500.0,
        aurora_max_cv=0.25,
        aurora_min_peak_nw=1.5,
        aurora_min_hot_fraction=0.001,
        tophat_radius_meters=2_500.0,
    )

    for name, (lat, lon) in BENCHMARK_SCENES.items():
        print(f"\n{'='*70}\nBenchmarking: {name} [{lat:.4f}, {lon:.4f}]")
        grid_info = aligner.get_master_grid_info(lat=lat, lon=lon)

        raw_nl = fetcher.fetch_raw_nightlight(grid_info=grid_info, year=year)
        field = fetcher.fetch_nightlight_potential(grid_info=grid_info, year=year, return_none_on_aurora=False)

        is_aurora_suppressed = bool(np.allclose(field, fetcher.floor_value, atol=1e-5))

        print(f"  Raw Radiance  | Max: {raw_nl.max():.2f} nW/cm²/sr | Mean: {raw_nl.mean():.4f}")
        print(f"  Potential     | Min: {field.min():.4f} | Max: {field.max():.4f} | Mean: {field.mean():.4f}")
        print(f"  Status        | {'AURORA SUPPRESSED (Floor Applied)' if is_aurora_suppressed else 'POINT SOURCES ACTIVE'}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"{name}\n[{lat:.4f}, {lon:.4f}] — {'Suppressed (Aurora)' if is_aurora_suppressed else 'Active Signal'}",
            fontsize=11,
            fontweight="bold",
            color="crimson" if is_aurora_suppressed else "darkgreen",
        )

        im0 = axes[0].imshow(raw_nl, cmap="inferno", origin="upper", vmin=0.0, vmax=max(raw_nl.max(), 1.0))
        axes[0].set_title(f"Raw VIIRS DNB Radiance\nMax: {raw_nl.max():.2f} nW", fontsize=10)
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(field, cmap="Greys" if is_aurora_suppressed else "magma", origin="upper", vmin=0.0, vmax=fetcher.alpha)
        axes[1].set_title(f"Activity Potential W(x, y)\nRange: [{field.min():.4f} – {field.max():.4f}]", fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        vals = field.ravel()
        axes[2].hist(vals, bins=50, color="#aaaaaa" if is_aurora_suppressed else "#d95f02", edgecolor="black", alpha=0.85, log=True)
        axes[2].axvline(fetcher.floor_value, color="blue", ls="--", lw=1.5, label=f"Floor ({fetcher.floor_value:.0e})")
        axes[2].axvline(float(np.median(vals)), color="green", ls=":", lw=1.5, label=f"Median ({np.median(vals):.4f})")
        axes[2].set_title("Pixel Value Distribution (Log Scale)", fontsize=10)
        axes[2].set_xlabel("Potential Value")
        axes[2].set_ylabel("Pixel Count")
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.show()