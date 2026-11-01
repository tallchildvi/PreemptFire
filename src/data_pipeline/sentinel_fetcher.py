from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from src.config import MAX_INVALID_RATIO, PIXEL_SCALE_METERS
from src.processing.grid_aligner import GridAligner


class SentinelFetcher:
    """Fetches raw COG-optimized Sentinel-2 L2A (T0 only) and Sentinel-1 RTC SAR aligned to the 100m master grid."""

    SCL_SNOW = [11]
    SCL_INVALID = [0, 1, 2, 3, 8, 9, 10, 11]

    OPTICAL_BANDS = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"
    ]
    REQUIRED_SPECTRAL_BANDS = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
    ]

    def __init__(
        self,
        max_nodata_fraction: float = MAX_INVALID_RATIO,
        max_workers: int = 10,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.max_nodata_fraction = max_nodata_fraction
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.aligner = GridAligner()

        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

    @classmethod
    def compute_loss_mask_from_scl(cls, scl_array: np.ndarray) -> np.ndarray:
        """Computes binary invalid pixel mask (clouds, shadows, nodata; snow is treated as valid)."""
        scl_clean = np.nan_to_num(scl_array, nan=0).astype(np.uint8)
        raw_invalid = np.isin(scl_clean, cls.SCL_INVALID)
        snow_mask = np.isin(scl_clean, cls.SCL_SNOW)
        return np.where(snow_mask, 0.0, raw_invalid).astype(np.float32)

    def _read_band_window(
        self,
        asset_url: str,
        bbox_wgs84: List[float],
        is_categorical: bool = False,
    ) -> Tuple[np.ndarray, Affine, Any]:
        """Streams a BBox window from remote COG overviews matching 2x Nyquist rate of the master grid."""
        try:
            signed_url = pc.sign(asset_url)
        except Exception:
            signed_url = asset_url

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                with rasterio.open(signed_url) as src:
                    west, south, east, north = bbox_wgs84
                    left, bottom, right, top = transform_bounds(
                        "EPSG:4326", src.crs, west, south, east, north
                    )
                    window = from_bounds(left, bottom, right, top, src.transform)

                    win_w = max(1.0, float(window.width))
                    win_h = max(1.0, float(window.height))

                    src_res_x = abs(src.transform.a)
                    if src.crs and src.crs.is_geographic:
                        src_res_x *= 111320.0

                    target_overview_res = max(src_res_x, PIXEL_SCALE_METERS * 0.5)
                    scale_factor = max(1.0, target_overview_res / max(src_res_x, 1e-6))

                    out_w = max(16, int(round(win_w / scale_factor)))
                    out_h = max(16, int(round(win_h / scale_factor)))

                    read_resampling = Resampling.nearest if is_categorical else Resampling.average

                    data = src.read(
                        1,
                        window=window,
                        out_shape=(out_h, out_w),
                        resampling=read_resampling,
                        boundless=True,
                        fill_value=0 if is_categorical else np.nan,
                    )

                    base_win_transform = src.window_transform(window)
                    win_transform = base_win_transform * Affine.scale(win_w / out_w, win_h / out_h)

                    return data.astype(np.float32), win_transform, src.crs

            except Exception as e:
                last_exc = e
                wait = self.retry_backoff ** attempt
                print(f"  [Sentinel Read] attempt {attempt + 1}/{self.max_retries} failed ({e}), retrying in {wait:.0f}s")
                time.sleep(wait)
                if attempt > 0:
                    try:
                        signed_url = pc.sign(asset_url)
                    except Exception:
                        pass

        raise RuntimeError(f"_read_band_window failed after {self.max_retries} attempts") from last_exc

    def _fetch_mosaicked_band(
        self, band_name: str, items: List[Any], grid_info: Dict
    ) -> Tuple[str, Optional[np.ndarray]]:
        """Reads and mosaics a single band across intersecting STAC items onto the master grid."""
        is_scl = band_name == "SCL"
        resampling = Resampling.nearest if is_scl else Resampling.average

        canvas_shape = grid_info["shape"]
        combined_canvas = (
            np.zeros(canvas_shape, dtype=np.uint8)
            if is_scl
            else np.full(canvas_shape, np.nan, dtype=np.float32)
        )

        for item in items:
            if band_name not in item.assets:
                continue

            try:
                asset_url = item.assets[band_name].href
                raw_data, src_transform, src_crs = self._read_band_window(
                    asset_url, grid_info["bbox_wgs84"], is_categorical=is_scl
                )

                if not is_scl and band_name in self.OPTICAL_BANDS:
                    raw_data = np.where((raw_data > 0) & np.isfinite(raw_data), raw_data, np.nan)
                    baseline = str(item.properties.get("s2:processing_baseline", "02.00"))
                    if baseline >= "04.00":
                        raw_data = np.maximum(raw_data - 1000.0, 1.0)
                    raw_data = raw_data / 10000.0

                aligned = self.aligner.align_raster_to_master(
                    src_array=raw_data,
                    src_crs=src_crs,
                    src_transform=src_transform,
                    grid_info=grid_info,
                    resampling_method=resampling,
                    dst_nodata=0.0 if is_scl else np.nan,
                )

                if is_scl:
                    valid_mask = aligned != 0
                    combined_canvas[valid_mask] = aligned[valid_mask].astype(np.uint8)
                else:
                    valid_mask = np.isfinite(aligned) & (aligned > 0.0)
                    combined_canvas[valid_mask] = aligned[valid_mask]

            except Exception as e:
                print(f"  [Warning] failed reading tile {item.id} for band {band_name}: {e}")
                continue

        return band_name, combined_canvas

    def _validate_optical_bands(self, aligned_bands: Dict[str, np.ndarray], expected_shape: Tuple[int, int]) -> bool:
        """Strictly validates that all 10 raw optical bands and SCL exist and meet quality thresholds."""
        if "SCL" not in aligned_bands or aligned_bands["SCL"] is None:
            return False

        min_valid_fraction = 1.0 - self.max_nodata_fraction
        for band in self.REQUIRED_SPECTRAL_BANDS:
            arr = aligned_bands.get(band)
            if arr is None or arr.shape != expected_shape:
                print(f"  [Validation Failed] Missing or malformed optical band: {band}")
                return False

            valid_mask = np.isfinite(arr) & (arr > 0.0)
            valid_ratio = float(valid_mask.mean())
            if valid_ratio < min_valid_fraction:
                print(
                    f"  [Validation Failed] Band {band} valid ratio {valid_ratio:.2f} "
                    f"is below threshold {min_valid_fraction:.2f}"
                )
                return False

            if float(np.nanstd(arr)) < 1e-6:
                print(f"  [Validation Failed] Band {band} has zero variance")
                return False

        return True

    def fetch_sentinel2_scene(
        self,
        grid_info: Dict,
        target_date_str: str,
        lookback_days: int = 30,
        max_cloud_cover: float = 60.0,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, str]:
        """Fetches the latest cloud-free Sentinel-2 L2A scene within [target_date - lookback_days, target_date]."""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)

        bbox = grid_info["bbox_wgs84"]
        date_range = f"{start_dt.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}"

        try:
            search = self.stac_client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                sortby=[{"field": "properties.datetime", "direction": "desc"}],
            )
            items = list(search.items())
        except Exception as e:
            print(f"  [Sentinel-2] STAC search error: {e}")
            return {}, np.array([]), ""

        if not items:
            return {}, np.array([]), ""

        items_by_date: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            d_str = item.datetime.strftime("%Y-%m-%d")
            items_by_date[d_str].append(item)

        sorted_dates = sorted(items_by_date.keys(), reverse=True)

        for candidate_date in sorted_dates:
            date_items = items_by_date[candidate_date]

            _, mosaicked_scl = self._fetch_mosaicked_band("SCL", date_items, grid_info)
            if mosaicked_scl is None:
                continue

            loss_mask = self.compute_loss_mask_from_scl(mosaicked_scl)
            if float(loss_mask.mean()) > self.max_nodata_fraction:
                continue

            aligned_bands: Dict[str, np.ndarray] = {}
            remaining_bands = [b for b in self.OPTICAL_BANDS if b != "SCL"]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(self._fetch_mosaicked_band, band, date_items, grid_info): band
                    for band in remaining_bands
                }
                for future in as_completed(future_map):
                    try:
                        b_name, b_arr = future.result()
                        if b_arr is not None:
                            aligned_bands[b_name] = b_arr
                    except Exception:
                        continue

            aligned_bands["SCL"] = mosaicked_scl

            if not self._validate_optical_bands(aligned_bands, grid_info["shape"]):
                continue

            return aligned_bands, loss_mask, candidate_date

        return {}, np.array([]), ""

    def fetch_sentinel1_sar(
        self, grid_info: Dict, target_date_str: str, lookback_days: int = 30
    ) -> Dict[str, np.ndarray]:
        """Fetches the latest raw Sentinel-1 RTC SAR (VV, VH) scene prior to target_date_str."""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)

        try:
            search = self.stac_client.search(
                collections=["sentinel-1-rtc"],
                bbox=grid_info["bbox_wgs84"],
                datetime=f"{start_dt.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}",
            )
            items = list(search.items())
        except Exception:
            return {}

        if not items:
            return {}

        items_by_date: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            d_str = item.datetime.strftime("%Y-%m-%d")
            items_by_date[d_str].append(item)

        sorted_dates = sorted(items_by_date.keys(), reverse=True)

        for d_str in sorted_dates:
            date_items = items_by_date[d_str]

            with ThreadPoolExecutor(max_workers=2) as pool:
                f_vv = pool.submit(self._fetch_mosaicked_band, "vv", date_items, grid_info)
                f_vh = pool.submit(self._fetch_mosaicked_band, "vh", date_items, grid_info)
                _, vv_canvas = f_vv.result()
                _, vh_canvas = f_vh.result()

            sar_bands: Dict[str, np.ndarray] = {}
            if vv_canvas is not None and np.isfinite(vv_canvas).mean() >= 0.50:
                sar_bands["SAR_VV"] = vv_canvas
            if vh_canvas is not None and np.isfinite(vh_canvas).mean() >= 0.50:
                sar_bands["SAR_VH"] = vh_canvas

            if len(sar_bands) == 2:
                return sar_bands

        return {}

    def fetch_all_radar_optical(self, lat: float, lon: float, target_date: str) -> Dict[str, Any]:
        """Fetches pre-fire T0 raw Sentinel-2 optical bands and Sentinel-1 SAR concurrently."""
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        pre_fire_dt = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)
        pre_fire_date = pre_fire_dt.strftime("%Y-%m-%d")

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_s2 = pool.submit(self.fetch_sentinel2_scene, grid_info, pre_fire_date, 30)
            f_sar = pool.submit(self.fetch_sentinel1_sar, grid_info, pre_fire_date, 30)

            bands_t0, loss_mask, t0_date = f_s2.result()
            sar_bands = f_sar.result()

        if not bands_t0 or not t0_date:
            return {}

        return {
            "grid_info": grid_info,
            "t0_date": t0_date,
            "bands_t0": bands_t0,
            "loss_mask": loss_mask,
            "sar_bands": sar_bands,
        }


if __name__ == "__main__":
    fetcher = SentinelFetcher()
    test_lat, test_lon = 41.6986, -7.0011
    test_date = "2022-08-09"

    print(f"Fetching raw 100m COG Sentinel data for [{test_lat}, {test_lon}] on {test_date}...")
    t_start = time.perf_counter()
    result = fetcher.fetch_all_radar_optical(lat=test_lat, lon=test_lon, target_date=test_date)

    if not result:
        print("[Error] Failed to find a valid cloud-free Sentinel-2 T0 scene.")
    else:
        elapsed = time.perf_counter() - t_start
        print(f"Successfully fetched in {elapsed:.2f}s | T0 Date: {result['t0_date']}")
        print(f"Optical bands: {list(result['bands_t0'].keys())} | SAR bands: {list(result['sar_bands'].keys())}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()

        rgb = np.stack([
            result["bands_t0"]["B04"],
            result["bands_t0"]["B03"],
            result["bands_t0"]["B02"],
        ], axis=-1)
        rgb = np.clip(np.nan_to_num(rgb, nan=0.0) * 3.5, 0.0, 1.0)
        axes[0].imshow(rgb)
        axes[0].set_title(f"Sentinel-2 RGB ({result['t0_date']})\nShape: {rgb.shape[:2]}")
        axes[0].axis("off")

        plot_items = [
            ("B08 (NIR)", result["bands_t0"]["B08"], "viridis"),
            ("B11 (SWIR1)", result["bands_t0"]["B11"], "magma"),
            ("B12 (SWIR2)", result["bands_t0"]["B12"], "inferno"),
            ("Raw SCL (uint8)", result["bands_t0"]["SCL"], "tab20"),
        ]
        if "SAR_VV" in result["sar_bands"]:
            plot_items.append(("Sentinel-1 SAR_VV", result["sar_bands"]["SAR_VV"], "plasma"))
        else:
            plot_items.append(("Loss Mask", result["loss_mask"], "gray"))

        for idx, (title, arr, cmap) in enumerate(plot_items, start=1):
            disp = np.nan_to_num(arr, nan=0.0)
            im = axes[idx].imshow(disp, cmap=cmap, origin="upper")
            axes[idx].set_title(f"{title}\nRange: [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.suptitle(f"SentinelFetcher Raw 100m COG Verification [{test_lat}, {test_lon}]", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()