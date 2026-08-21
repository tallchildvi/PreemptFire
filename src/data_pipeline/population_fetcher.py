import os
from typing import Optional

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
from scipy.signal import fftconvolve

from src.processing.grid_aligner import GridAligner


class PopulationFetcher:
    def __init__(
        self,
        project_id: Optional[str] = None,
        buffer_meters: float = 50_000.0,
        decay_distance_meters: float = 8_000.0,
        long_decay_meters: float = 30_000.0,
        short_weight: float = 0.75,
        long_weight: float = 0.25,
        floor_value: float = 1e-3,
        reference_population: float = 1_000_000.0,
        tail_exponent: float = 2.0,
    ):
        load_dotenv()
        self.aligner = GridAligner()
        self.project_id = project_id or os.getenv("GEE_PROJECT")

        self.buffer_meters = buffer_meters
        self.decay_distance_meters = decay_distance_meters
        self.long_decay_meters = long_decay_meters
        self.short_weight = short_weight
        self.long_weight = long_weight
        self.floor_value = floor_value
        self.reference_population = reference_population
        self.tail_exponent = tail_exponent

        if not np.isclose(short_weight + long_weight, 1.0):
            raise ValueError("short_weight + long_weight must equal 1.0")

        self._init_earth_engine()
        self.collection_primary = "projects/sat-io/open-datasets/WORLDPOP/pop"
        self.collection_fallback = "WorldPop/GP/100m/pop"

    def _init_earth_engine(self):
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Earth Engine: {e}\n"
                f"Run 'ee.Authenticate()' or set GEE_PROJECT in .env"
            )

    def _get_population_image(self, year: int) -> ee.Image:
        try:
            col = ee.ImageCollection(self.collection_primary)
            filtered = col.filter(ee.Filter.calendarRange(year, year, "year"))
            if filtered.size().getInfo() > 0:
                return filtered.select(["population"]).mosaic()
        except Exception:
            pass

        col_official = ee.ImageCollection(self.collection_fallback)
        filtered_official = col_official.filter(ee.Filter.calendarRange(year, year, "year"))
        if filtered_official.size().getInfo() > 0:
            return filtered_official.select(["population"]).mosaic()

        return col_official.select(["population"]).sort("system:time_start", False).first()

    def fetch_raw_population(
        self,
        grid_info: dict,
        year: int = 2020,
        apply_log1p: bool = False,
    ) -> np.ndarray:
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

        pop_image = self._get_population_image(year=year)
        try:
            download_url = pop_image.getDownloadURL({
                "region": region,
                "scale": 100,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            })
            resp = requests.get(download_url, timeout=40)
            resp.raise_for_status()
        except Exception as e:
            print(f"[PopulationFetcher] Raw fetch failed: {e}")
            return np.zeros(shape, dtype=np.float32)

        raw_pop_10m = np.zeros(shape, dtype=np.float32)
        with MemoryFile(resp.content) as memfile:
            with memfile.open() as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=raw_pop_10m,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=master_crs,
                    dst_transform=master_transform,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=0.0,
                )

        raw_pop_10m = np.nan_to_num(raw_pop_10m, nan=0.0)
        raw_pop_10m = np.clip(raw_pop_10m, 0.0, None)

        if apply_log1p:
            return np.log1p(raw_pop_10m).astype(np.float32)
        return raw_pop_10m.astype(np.float32)

    @staticmethod
    def _make_exp_kernel(radius_px: int, decay_m: float, pixel_m: float = 100.0) -> np.ndarray:
        y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        dist_m = np.sqrt(x**2 + y**2) * pixel_m
        return np.exp(-dist_m / decay_m).astype(np.float32)

    @staticmethod
    def _make_power_kernel(radius_px: int, scale_m: float, exponent: float = 2.0, pixel_m: float = 100.0) -> np.ndarray:
        y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        dist_m = np.sqrt(x**2 + y**2) * pixel_m
        return (1.0 / (1.0 + dist_m / scale_m) ** exponent).astype(np.float32)

    def fetch_population(self, grid_info: dict, year: int = 2020) -> np.ndarray:
        shape = grid_info["shape"]
        master_crs = grid_info["crs"]
        master_transform = grid_info["transform"]
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]

        b_min_x, b_max_x = min_x - self.buffer_meters, max_x + self.buffer_meters
        b_min_y, b_max_y = min_y - self.buffer_meters, max_y + self.buffer_meters

        transformer = Transformer.from_crs(master_crs, "EPSG:4326", always_xy=True)
        corners = [(b_min_x, b_min_y), (b_min_x, b_max_y), (b_max_x, b_min_y), (b_max_x, b_max_y)]
        corners_wgs84 = [transformer.transform(x, y) for x, y in corners]
        b_lons = [pt[0] for pt in corners_wgs84]
        b_lats = [pt[1] for pt in corners_wgs84]

        region = ee.Geometry.Rectangle(
            [min(b_lons), min(b_lats), max(b_lons), max(b_lats)],
            proj="EPSG:4326",
            geodesic=False,
        )

        pop_image = self._get_population_image(year=year)
        try:
            download_url = pop_image.getDownloadURL({
                "region": region,
                "scale": 100,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            })
            resp = requests.get(download_url, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"[PopulationFetcher] GEE population fetch failed: {e}")
            return np.full(shape, self.floor_value, dtype=np.float32)

        pop_pressure_10m = np.zeros(shape, dtype=np.float32)

        with MemoryFile(resp.content) as memfile:
            with memfile.open() as src:
                raw_100m = src.read(1).astype(np.float32)
                raw_100m = np.nan_to_num(raw_100m, nan=0.0)
                raw_100m = np.clip(raw_100m, 0.0, None)

                pixel_m = 100.0
                short_r = int(round(self.decay_distance_meters * 4.0 / pixel_m))
                k_short = self._make_exp_kernel(short_r, self.decay_distance_meters, pixel_m)

                raster_diag_m = np.sqrt((raw_100m.shape[1] * pixel_m)**2 + (raw_100m.shape[0] * pixel_m)**2)
                long_r = int(np.ceil(raster_diag_m / 2.0 / pixel_m))
                k_long = self._make_power_kernel(long_r, self.long_decay_meters, self.tail_exponent, pixel_m)

                raw_padded_short = np.pad(raw_100m, short_r, mode="reflect")
                raw_padded_long = np.pad(raw_100m, long_r, mode="reflect")

                conv_short = fftconvolve(raw_padded_short, k_short, mode="same")
                conv_short = conv_short[short_r:-short_r or None, short_r:-short_r or None]

                conv_long = fftconvolve(raw_padded_long, k_long, mode="same")
                conv_long = conv_long[long_r:-long_r or None, long_r:-long_r or None]

                blended = (self.short_weight * conv_short) + (self.long_weight * conv_long)
                blended = np.clip(blended, 0.0, None)

                log_anchor = np.log1p(float(self.reference_population))
                potential_100m = (np.log1p(blended) / log_anchor).astype(np.float32)

                reproject(
                    source=potential_100m,
                    destination=pop_pressure_10m,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=master_crs,
                    dst_transform=master_transform,
                    resampling=Resampling.bilinear,
                    dst_nodata=0.0,
                )

        pop_pressure_10m = np.nan_to_num(pop_pressure_10m, nan=0.0, posinf=0.0, neginf=0.0)
        pop_pressure_10m = np.clip(pop_pressure_10m, 0.0, None)
        pop_pressure_10m = np.maximum(pop_pressure_10m, self.floor_value)

        return pop_pressure_10m.astype(np.float32)


if __name__ == "__main__":
    aligner = GridAligner()

    # lat, lon = 59.3862126, -108.8931627
    lat, lon = 69.9286119, -127.4445177
    # lat, lon = 53.8920497, -113.2228868
    # lat, lon = 53.5461000, -113.4937000
    # lat, lon = 62.6787906, -136.6717606
    # lat, lon = 50.8912718, -120.4779282
    # lat, lon = 49.4220429, -123.5368219

    print(f"{lat}, {lon}")
    grid_info = aligner.get_master_grid_info(lat=lat, lon=lon)

    fetcher = PopulationFetcher(
        buffer_meters=50_000.0,
        decay_distance_meters=8_000.0,
        long_decay_meters=30_000.0,
        short_weight=0.75,
        long_weight=0.25,
        floor_value=1e-3,
        reference_population=1_000_000.0,
        tail_exponent=2.0,
    )

    raw_pop = fetcher.fetch_raw_population(grid_info=grid_info, year=2023)
    pop_field = fetcher.fetch_population(grid_info=grid_info, year=2023)
    log_anchor = np.log1p(fetcher.reference_population)

    print("\n--- GEE WorldPop Population Results ---")
    print(f"Raw Population  | sum: {raw_pop.sum():.1f} persons total in scene")
    print(f"                | max: {raw_pop.max():.2f} persons/100m-cell")
    print(f"Log anchor      | log1p({fetcher.reference_population:.0f}) = {log_anchor:.3f}")
    print(f"Smooth Field    | Min: {pop_field.min():.4f} | Max: {pop_field.max():.4f}")
    print(f"                | Non-floor pixels: {(pop_field > fetcher.floor_value * 1.01).mean() * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im0 = axes[0].imshow(np.log1p(raw_pop), cmap="inferno", origin="upper")
    axes[0].set_title(f"Raw Discrete Population ({lat}, {lon}): {raw_pop.max():.1f} persons/cell", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pop_field, cmap="magma", origin="upper", vmin=0.0, vmax=max(pop_field.max(), 0.15))
    axes[1].set_title(
        f"Multi-scale Potential (exp {fetcher.decay_distance_meters / 1e3:.0f} km + "
        f"power-law α={fetcher.tail_exponent} scale={fetcher.long_decay_meters / 1e3:.0f} km)\n"
        f"Range: [{pop_field.min():.4f} – {pop_field.max():.4f}] (1.0 = {fetcher.reference_population / 1e6:.0f} M ref)",
        fontsize=11, fontweight="bold",
    )
    axes[1].axis("off")
    cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label(f"potential [ref = {int(fetcher.reference_population):,} persons → 1.0]")

    plt.tight_layout()
    plt.show()