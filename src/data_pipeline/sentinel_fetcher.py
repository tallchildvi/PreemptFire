from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from src.processing.grid_aligner import GridAligner


class SentinelFetcher:
    
    SCL_WATER = [6]
    SCL_SNOW = [11]
    SCL_CLOUDS = [8, 9, 10]
    SCL_CLOUD_SHADOWS = [3]
    SCL_INVALID = [0, 1, 2, 3, 8, 9, 10, 11]

    OPTICAL_BANDS = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"
    ]

    def __init__(self, max_nodata_fraction: float = 0.20, max_workers: int = 10):
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        self.aligner = GridAligner()
        self.max_nodata_fraction = max_nodata_fraction
        self.max_workers = max_workers

    @staticmethod
    def extract_scl_masks(scl_array: np.ndarray) -> Dict[str, np.ndarray]:
        scl_clean = np.nan_to_num(scl_array, nan=0).astype(np.uint8)
        return {
            "MASK_WATER": np.isin(scl_clean, SentinelFetcher.SCL_WATER).astype(np.float32),
            "MASK_SNOW": np.isin(scl_clean, SentinelFetcher.SCL_SNOW).astype(np.float32),
            "MASK_CLOUDS": np.isin(scl_clean, SentinelFetcher.SCL_CLOUDS).astype(np.float32),
            "MASK_CLOUD_SHADOWS": np.isin(scl_clean, SentinelFetcher.SCL_CLOUD_SHADOWS).astype(np.float32),
            "MASK_INVALID": np.isin(scl_clean, SentinelFetcher.SCL_INVALID).astype(np.float32),
        }

    def _read_band_window(
        self, asset_url: str, bbox_wgs84: List[float]
    ) -> Tuple[np.ndarray, Affine, Any]:
        """Signs URL dynamically right before fetching to eliminate SAS token expiration."""
        signed_url = pc.sign(asset_url)
        with rasterio.open(signed_url) as src:
            west, south, east, north = bbox_wgs84
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, west, south, east, north
            )
            window = from_bounds(left, bottom, right, top, src.transform)
            data = src.read(1, window=window, boundless=True, fill_value=np.nan)
            win_transform = src.window_transform(window)
            return data.astype(np.float32), win_transform, src.crs

    def _fetch_single_band(
        self, band_name: str, item: Any, grid_info: Dict
    ) -> Tuple[str, Optional[np.ndarray]]:
        if band_name not in item.assets:
            return band_name, None

        try:
            asset_url = item.assets[band_name].href
            raw_data, src_transform, src_crs = self._read_band_window(
                asset_url, grid_info["bbox_wgs84"]
            )

            resampling = Resampling.nearest if band_name == "SCL" else Resampling.bilinear
            dst_nodata = 0 if band_name == "SCL" else np.nan

            aligned = self.aligner.align_raster_to_master(
                src_array=raw_data,
                src_crs=src_crs,
                src_transform=src_transform,
                grid_info=grid_info,
                resampling_method=resampling,
                dst_nodata=dst_nodata,
            )
            return band_name, aligned
        except Exception as e:
            print(f"  [Warning] Failed band {band_name}: {e}")
            return band_name, None

    def fetch_sentinel2_scene(
        self,
        grid_info: Dict,
        target_date_str: str,
        lookback_days: int = 15,
        max_cloud_cover: float = 25.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
        """Searches and selects a valid scene where NoData < 20%, parallel fetching all bands."""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)

        bbox = grid_info["bbox_wgs84"]
        date_range = f"{start_dt.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}"

        search = self.stac_client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
        )
        items = list(search.items())
        if not items:
            return {}, {}, ""

        for candidate_item in items:
            if "SCL" not in candidate_item.assets:
                continue

            _, scl_check = self._fetch_single_band("SCL", candidate_item, grid_info)
            if scl_check is None:
                continue

            nodata_ratio = float((scl_check == 0).mean())
            if nodata_ratio > self.max_nodata_fraction:
                print(f"  [Skip] Scene {candidate_item.id} rejected: NoData is {nodata_ratio * 100:.1f}% (> 20%)")
                continue

            actual_date = candidate_item.datetime.strftime("%Y-%m-%d")

            aligned_bands = {"SCL": scl_check}
            remaining_bands = [b for b in self.OPTICAL_BANDS if b != "SCL"]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._fetch_single_band, band, candidate_item, grid_info)
                    for band in remaining_bands
                ]
                for f in futures:
                    b_name, b_arr = f.result()
                    if b_arr is not None:
                        aligned_bands[b_name] = b_arr

            scl_array = aligned_bands.pop("SCL")
            masks = self.extract_scl_masks(scl_array)

            return aligned_bands, masks, actual_date

        print(f"  [Error] No S2 scene found with < {self.max_nodata_fraction*100:.0f}% NoData.")
        return {}, {}, ""

    def fetch_sentinel1_sar(
        self, grid_info: Dict, target_date_str: str, window_days: int = 12
    ) -> Dict[str, np.ndarray]:
        """Fetches S1 RTC SAR with fresh SAS signing."""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=window_days)
        end_dt = target_dt + timedelta(days=window_days)

        search = self.stac_client.search(
            collections=["sentinel-1-rtc"],
            bbox=grid_info["bbox_wgs84"],
            datetime=f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}",
        )
        items = list(search.items())
        if not items:
            return {}

        for item in items:
            sar_bands = {}
            for pol in ["vv", "vh"]:
                if pol not in item.assets:
                    continue
                try:
                    asset_url = item.assets[pol].href
                    raw_data, src_transform, src_crs = self._read_band_window(
                        asset_url, grid_info["bbox_wgs84"]
                    )
                    sar_bands[f"SAR_{pol.upper()}"] = self.aligner.align_raster_to_master(
                        src_array=raw_data,
                        src_crs=src_crs,
                        src_transform=src_transform,
                        grid_info=grid_info,
                        resampling_method=Resampling.bilinear,
                        dst_nodata=np.nan,
                    )
                except Exception as e:
                    print(f"  [Warning] S1 SAR {pol} fetch failed: {e}")
                    break

            if len(sar_bands) == 2:
                return sar_bands

        return {}

    def fetch_all_radar_optical(self, lat: float, lon: float, target_date: str) -> Dict[str, Any]:
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        # 1. Fetch T0
        bands_t0, masks_t0, t0_date = self.fetch_sentinel2_scene(
            grid_info, target_date, lookback_days=15
        )
        if not bands_t0:
            return {}

        # 2. Fetch Tprev
        t0_dt = datetime.strptime(t0_date, "%Y-%m-%d")
        tprev_target = (t0_dt - timedelta(days=10)).strftime("%Y-%m-%d")
        bands_tprev, masks_tprev, tprev_date = self.fetch_sentinel2_scene(
            grid_info, tprev_target, lookback_days=30
        )

        # 3. Fetch S1 SAR
        sar_bands = self.fetch_sentinel1_sar(grid_info, t0_date)

        return {
            "grid_info": grid_info,
            "t0_date": t0_date,
            "tprev_date": tprev_date,
            "bands_t0": bands_t0,
            "masks_t0": masks_t0,
            "bands_tprev": bands_tprev,
            "masks_tprev": masks_tprev,
            "sar_bands": sar_bands,
        }