import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.transform import Affine

from src.processing.grid_aligner import GridAligner
from src.config import COARSE, PIXEL_SCALE_METERS

class SentinelFetcher:
    """fetches multi-tile mosaicked sentinel-2 l2a and sentinel-1 rtc sar data aligned to master grid."""

    SCL_WATER = [6]
    SCL_SNOW = [11]
    SCL_CLOUDS = [8, 9, 10]
    SCL_CLOUD_SHADOWS = [3]
    SCL_INVALID = [0, 1, 2, 3, 8, 9, 10, 11]

    OPTICAL_BANDS = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"
    ]

    def __init__(
        self,
        max_nodata_fraction: float = 0.20,
        max_workers: int = 10,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.max_nodata_fraction = max_nodata_fraction
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.aligner = GridAligner()

        print("    [Sentinel] Initializing STAC Client...")
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

    @classmethod
    def extract_scl_masks(cls, scl_array: np.ndarray) -> Dict[str, np.ndarray]:
        scl_clean = np.nan_to_num(scl_array, nan=0).astype(np.uint8)
        return {
            "MASK_WATER": np.isin(scl_clean, cls.SCL_WATER).astype(np.float32),
            "MASK_SNOW": np.isin(scl_clean, cls.SCL_SNOW).astype(np.float32),
            "MASK_CLOUDS": np.isin(scl_clean, cls.SCL_CLOUDS).astype(np.float32),
            "MASK_CLOUD_SHADOWS": np.isin(scl_clean, cls.SCL_CLOUD_SHADOWS).astype(np.float32),
            "MASK_INVALID": np.isin(scl_clean, cls.SCL_INVALID).astype(np.float32),
        }

    def _read_band_window(
        self,
        asset_url: str,
        bbox_wgs84: List[float],
        resampling: Resampling = Resampling.average,
        native: bool = False,
    ) -> Tuple[np.ndarray, Affine, Any]:

        env = rasterio.Env(
            GDAL_HTTP_TIMEOUT="15",
            GDAL_HTTP_MAX_RETRY="3",
            GDAL_HTTP_RETRY_DELAY="1",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_USE_IPV6="NO",          
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", 
            VSI_CACHE="TRUE",                     
            GDAL_HTTP_MULTIMAC="YES",             
        )

        try:
            signed_url = pc.sign(asset_url)
        except Exception:
            signed_url = asset_url

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                with env:
                    with rasterio.open(signed_url) as src:
                        west, south, east, north = bbox_wgs84
                        left, bottom, right, top = transform_bounds(
                            "EPSG:4326", src.crs, west, south, east, north
                        )
                        window = from_bounds(left, bottom, right, top, src.transform)
                        win_transform = src.window_transform(window)
                        if COARSE and not native:
                            k = PIXEL_SCALE_METERS / abs(src.res[0])
                            out_h = max(1, int(round(window.height / k)))
                            out_w = max(1, int(round(window.width / k)))
                            data = src.read(
                                1, window=window, out_shape=(out_h, out_w),
                                boundless=True, fill_value=np.nan, resampling=resampling,
                            )
                            win_transform = win_transform * Affine.scale(
                                window.width / out_w, window.height / out_h
                            )
                        else:
                            data = src.read(1, window=window, boundless=True, fill_value=np.nan)
                        return data.astype(np.float32), win_transform, src.crs

            except Exception as e:
                last_exc = e
                wait = self.retry_backoff ** attempt
                print(f"      [Read Warning] Attempt {attempt + 1}/{self.max_retries} failed for {signed_url[-40:]}... ({type(e).__name__}). Retrying in {wait}s")
                time.sleep(wait)
                if attempt > 0:
                    try:
                        signed_url = pc.sign(asset_url)
                    except Exception:
                        pass

        # Якщо всі ретраї провалилися, генеруємо помилку. _fetch_mosaicked_band її зловить.
        raise RuntimeError(f"Rasterio read failed after {self.max_retries} attempts: {last_exc}")

    def _fetch_mosaicked_band(
        self, band_name: str, items: List[Any], grid_info: Dict
    ) -> Tuple[str, Optional[np.ndarray]]:
        is_scl = (band_name == "SCL")
        resampling = Resampling.nearest if is_scl else Resampling.bilinear

        canvas_shape = grid_info["shape"]
        combined_canvas = np.zeros(canvas_shape, dtype=np.uint8) if is_scl else np.full(canvas_shape, np.nan, dtype=np.float32)

        for item in items:
            if band_name not in item.assets:
                continue

            try:
                asset_url = item.assets[band_name].href
                raw_data, src_transform, src_crs = self._read_band_window(
                    asset_url, grid_info["bbox_wgs84"],
                    resampling=Resampling.nearest if is_scl else Resampling.average,
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
                    valid_mask = (aligned != 0)
                    combined_canvas[valid_mask] = aligned[valid_mask].astype(np.uint8)
                else:
                    valid_mask = np.isfinite(aligned) & (aligned > 0.0)
                    combined_canvas[valid_mask] = aligned[valid_mask]

            except Exception as e:
                # Перехоплюємо RuntimeError з _read_band_window і прокидаємо його вище, 
                # щоб НЕ перебирати наступні тайли, якщо перший же тайл на цій даті битий.
                raise RuntimeError(f"Tile {item.id} is broken: {e}")

        return band_name, combined_canvas

    def _scl_fraction_masks(self, items: List[Any], grid_info: Dict) -> Dict[str, np.ndarray]:
        groups = {
            "MASK_WATER": self.SCL_WATER,
            "MASK_SNOW": self.SCL_SNOW,
            "MASK_CLOUDS": self.SCL_CLOUDS,
            "MASK_CLOUD_SHADOWS": self.SCL_CLOUD_SHADOWS,
            "MASK_INVALID": self.SCL_INVALID,
        }
        shape = grid_info["shape"]
        num = {k: np.zeros(shape, np.float32) for k in groups}
        den = np.zeros(shape, np.float32)

        for item in items:
            if "SCL" not in item.assets:
                continue
            try:
                raw, tr, crs = self._read_band_window(
                    item.assets["SCL"].href, grid_info["bbox_wgs84"], native=True
                )
            except Exception as e:
                print(f"      [Warning] SCL read failed for {item.id}: {e}")
                continue
            scl = np.nan_to_num(raw, nan=0).astype(np.uint8)
            cov = self.aligner.align_raster_to_master(
                (scl != 0).astype(np.float32), crs, tr, grid_info,
                Resampling.average, dst_nodata=0.0,
            )
            for k, cls in groups.items():
                frac = self.aligner.align_raster_to_master(
                    np.isin(scl, cls).astype(np.float32), crs, tr, grid_info,
                    Resampling.average, dst_nodata=0.0,
                )
                num[k] += frac * cov
            den += cov

        cov_c = np.clip(den, 0.0, 1.0)
        out: Dict[str, np.ndarray] = {}
        for k in groups:
            m = np.divide(num[k], den, out=np.zeros(shape, np.float32), where=den > 1e-6)
            out[k] = np.clip(m, 0.0, 1.0)

        out["MASK_INVALID"] = np.clip(out["MASK_INVALID"] * cov_c + (1.0 - cov_c), 0.0, 1.0)
        return out

    def fetch_sentinel2_scene(
        self,
        grid_info: Dict,
        target_date_str: str,
        lookback_days: int = 25,
        max_cloud_cover: float = 60.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)

        bbox = grid_info["bbox_wgs84"]
        date_range = f"{start_dt.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}"

        print(f"    [S2] Searching STAC from {date_range} for bbox {bbox}...")
        try:
            search = self.stac_client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                sortby=[{"field": "properties.datetime", "direction": "desc"}],
            )
            items = list(search.items())
            print(f"    [S2] STAC returned {len(items)} items.")
        except Exception as e:
            print(f"    [S2] STAC search failed: {e}")
            return {}, {}, ""

        if not items:
            return {}, {}, ""

        items_by_date: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            d_str = item.datetime.strftime("%Y-%m-%d")
            items_by_date[d_str].append(item)

        sorted_dates = sorted(items_by_date.keys(), reverse=True)

        for candidate_date in sorted_dates:
            date_items = items_by_date[candidate_date]
            print(f"    [S2] Checking SCL mask for date {candidate_date} (has {len(date_items)} tiles)...")

            try:
                _, mosaicked_scl = self._fetch_mosaicked_band("SCL", date_items, grid_info)
            except RuntimeError as e:
                print(f"    [S2] Date {candidate_date} skipped due to broken files: {e}")
                continue
                
            if mosaicked_scl is None:
                continue
            
            fast_nodata = float(np.isin(mosaicked_scl, [0, 1, 2, 3, 8, 9, 10]).mean())
            if fast_nodata > self.max_nodata_fraction + 0.05:
                print(f"    [S2] Date {candidate_date} rejected by FAST check (nodata={fast_nodata:.2f})")
                continue

            if COARSE:
                frac_masks = self._scl_fraction_masks(date_items, grid_info)
                nodata_ratio = float(frac_masks["MASK_INVALID"].mean())
            else:
                frac_masks = None
                nodata_ratio = fast_nodata

            if nodata_ratio > self.max_nodata_fraction:
                print(f"    [S2] Date {candidate_date} rejected (nodata={nodata_ratio:.2f})")
                continue
            
            print(f"    [S2] Date {candidate_date} ACCEPTED (nodata={nodata_ratio:.2f}). Fetching {len(self.OPTICAL_BANDS)-1} optical bands...")

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
                    except Exception as e:
                        print(f"    [S2] Failed fetching band: {e}")
                        continue

            aligned_bands["SCL"] = mosaicked_scl
            masks = frac_masks if COARSE else self.extract_scl_masks(mosaicked_scl)
            print(f"    [S2] All optical bands fetched for {candidate_date}.")
            return aligned_bands, masks, candidate_date

        print("    [S2] All candidate dates rejected due to cloud cover or broken files.")
        return {}, {}, ""


    def fetch_sentinel1_sar(
        self, grid_info: Dict, target_date_str: str, lookback_days: int = 30
    ) -> Dict[str, np.ndarray]:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)

        print(f"    [SAR] Searching STAC...")
        try:
            search = self.stac_client.search(
                collections=["sentinel-1-rtc"],
                bbox=grid_info["bbox_wgs84"],
                datetime=f"{start_dt.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}",
            )
            items = list(search.items())
            print(f"    [SAR] STAC returned {len(items)} items.")
        except Exception as e:
            print(f"    [SAR] STAC search failed: {e}")
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
            print(f"    [SAR] Fetching VH/VV for date {d_str} (Sequential)...")

            try:
                # Виконуємо послідовно! Якщо файл битий, генерується RuntimeError.
                _, vv_canvas = self._fetch_mosaicked_band("vv", date_items, grid_info)
                _, vh_canvas = self._fetch_mosaicked_band("vh", date_items, grid_info)

                sar_bands: Dict[str, np.ndarray] = {}
                if vv_canvas is not None and not np.isnan(vv_canvas).all():
                    sar_bands["SAR_VV"] = vv_canvas
                if vh_canvas is not None and not np.isnan(vh_canvas).all():
                    sar_bands["SAR_VH"] = vh_canvas

                if len(sar_bands) == 2:
                    print(f"    [SAR] Fetch completed successfully.")
                    return sar_bands
                
                # Якщо канали заповнені NaN (що дивно, бо помилки перехоплюються вище), виходимо
                print(f"    [SAR] Channels are empty for {d_str}. Aborting SAR search.")
                return {}

            except RuntimeError as e:
                # Якщо зловили RuntimeError (битий файл на Azure), миттєво припиняємо пошук SAR!
                # БІЛЬШЕ НЕМАЄ ПЕРЕБОРУ ДАТ!
                print(f"    [SAR] ABORTING: File is broken on Azure server ({e}).")
                return {}

        return {}

    def fetch_all_radar_optical(self, lat: float, lon: float, target_date: str) -> Dict[str, Any]:
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        pre_fire_dt = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)
        pre_fire_date = pre_fire_dt.strftime("%Y-%m-%d")

        print(f"  [Sentinel Fetcher] Starting Sentinel-2 optical extraction...")
        bands_t0, masks_t0, t0_date = self.fetch_sentinel2_scene(
            grid_info, pre_fire_date, lookback_days=25
        )
        if not bands_t0 or not t0_date:
            print("  [Sentinel Fetcher] ABORTING: S2 fetch failed.")
            return {}

        print(f"  [Sentinel Fetcher] Starting Sentinel-1 SAR extraction...")
        sar_bands = self.fetch_sentinel1_sar(grid_info, pre_fire_date, lookback_days=30)
        
        if not sar_bands:
            print("  [Sentinel Fetcher] WARNING: SAR data not found or broken. Saving scene with optical only.")

        return {
            "grid_info": grid_info,
            "t0_date": t0_date,
            "bands_t0": bands_t0,
            "masks_t0": masks_t0,
            "sar_bands": sar_bands,
        }