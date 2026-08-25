from datetime import datetime, timedelta
import math
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

from src.processing.grid_aligner import GridAligner


class SentinelFetcher:
    """Fetches Sentinel-2 (T0, Tprev) and Sentinel-1 SAR bands from Planetary Computer STAC."""

    SCL_WATER = [6]
    SCL_SNOW = [11]
    SCL_CLOUDS = [8, 9, 10]          # Medium/High probability clouds + Cirrus
    SCL_CLOUD_SHADOWS = [3]          # Cloud shadows
    SCL_INVALID = [0, 1, 2, 3, 8, 9, 10, 11]  # NoData, Defects, Shadows, Clouds, Snow
 
    def __init__(self):
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        self.aligner = GridAligner()

    @staticmethod
    def extract_scl_masks(scl_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts individual binary float32 masks (0.0 or 1.0) from raw SCL raster."""
        scl_clean = np.nan_to_num(scl_array, nan=0).astype(np.uint8)

        mask_water = np.isin(scl_clean, SentinelFetcher.SCL_WATER).astype(np.float32)
        mask_snow = np.isin(scl_clean, SentinelFetcher.SCL_SNOW).astype(np.float32)
        mask_clouds = np.isin(scl_clean, SentinelFetcher.SCL_CLOUDS).astype(np.float32)
        mask_cloud_shadows = np.isin(scl_clean, SentinelFetcher.SCL_CLOUD_SHADOWS).astype(np.float32)
        mask_invalid = np.isin(scl_clean, SentinelFetcher.SCL_INVALID).astype(np.float32)

        return {
            "MASK_WATER": mask_water,
            "MASK_SNOW": mask_snow,
            "MASK_CLOUDS": mask_clouds,
            "MASK_CLOUD_SHADOWS": mask_cloud_shadows,
            "MASK_INVALID": mask_invalid,  # Full mask for Loss calculation
        }

    def _read_band_window(
        self, asset_url: str, bbox_wgs84: List[float]
    ) -> Tuple[np.ndarray, Affine, Any]:
        """Reads a specific spatial window from a signed cloud COG URL."""
        with rasterio.open(asset_url) as src:
            west, south, east, north = bbox_wgs84
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, west, south, east, north
            )
            window = from_bounds(left, bottom, right, top, src.transform)
            data = src.read(1, window=window, boundless=True, fill_value=np.nan)
            win_transform = src.window_transform(window)
            return data.astype(np.float32), win_transform, src.crs

    def fetch_sentinel2_scene(
        self,
        grid_info: Dict,
        target_date_str: str,
        lookback_days: int = 15,
        max_cloud_cover: float = 20.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
        """Fetches optical bands and SCL-derived masks for the clearest scene."""
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

        selected_item = items[0]
        actual_date = selected_item.datetime.strftime("%Y-%m-%d")

        bands_to_fetch = [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"
        ]
        aligned_bands = {}

        for band in bands_to_fetch:
            if band not in selected_item.assets:
                continue
            asset_url = selected_item.assets[band].href
            raw_data, src_transform, src_crs = self._read_band_window(asset_url, bbox)

            resampling = Resampling.nearest if band == "SCL" else Resampling.bilinear
            dst_nodata = 0 if band == "SCL" else np.nan

            aligned_bands[band] = self.aligner.align_raster_to_master(
                src_array=raw_data,
                src_crs=src_crs,
                src_transform=src_transform,
                grid_info=grid_info,
                resampling_method=resampling,
                dst_nodata=dst_nodata,
            )

        scl_array = aligned_bands.pop("SCL")
        masks = self.extract_scl_masks(scl_array)

        return aligned_bands, masks, actual_date

    def fetch_sentinel1_sar(
        self,
        grid_info: Dict,
        target_date_str: str,
        window_days: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Fetches Sentinel-1 RTC SAR (VV, VH polarizations)."""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=window_days)
        end_dt = target_dt + timedelta(days=window_days)

        bbox = grid_info["bbox_wgs84"]
        date_range = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"

        search = self.stac_client.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=date_range,
        )
        items = list(search.items())
        if not items:
            return {}

        selected_item = items[0]
        sar_bands = {}

        for pol in ["vv", "vh"]:
            if pol not in selected_item.assets:
                continue
            asset_url = selected_item.assets[pol].href
            raw_data, src_transform, src_crs = self._read_band_window(asset_url, bbox)

            sar_bands[f"SAR_{pol.upper()}"] = self.aligner.align_raster_to_master(
                src_array=raw_data,
                src_crs=src_crs,
                src_transform=src_transform,
                grid_info=grid_info,
                resampling_method=Resampling.bilinear,
                dst_nodata=np.nan,
            )

        return sar_bands

    def fetch_all_radar_optical(self, lat: float, lon: float, target_date: str) -> Dict[str, Any]:
        """Orchestrates fetching T0 optical, T0 masks, Tprev optical, Tprev masks, and SAR."""
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        # 1. Fetch T0
        bands_t0, masks_t0, t0_date = self.fetch_sentinel2_scene(
            grid_info, target_date, lookback_days=15
        )
        if not bands_t0:
            print(f"Warning: No clear S2 T0 scene found for ({lat}, {lon}) near {target_date}")
            return {}

        # 2. Fetch Tprev
        t0_dt = datetime.strptime(t0_date, "%Y-%m-%d")
        tprev_target = (t0_dt - timedelta(days=10)).strftime("%Y-%m-%d")
        bands_tprev, masks_tprev, tprev_date = self.fetch_sentinel2_scene(
            grid_info, tprev_target, lookback_days=25
        )

        # 3. Fetch Sentinel-1 SAR
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