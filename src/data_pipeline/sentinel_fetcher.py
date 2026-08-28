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
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from src.processing.grid_aligner import GridAligner


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

        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

    # scl mask extraction

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

    # low-level band read

    def _read_band_window(
        self, asset_url: str, bbox_wgs84: List[float]
    ) -> Tuple[np.ndarray, Affine, Any]:
        """signs url and reads bounding box window with boundless padding."""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
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

            except Exception as e:
                last_exc = e
                wait = self.retry_backoff ** attempt
                print(f"  [Read] attempt {attempt + 1}/{self.max_retries} failed ({e}), retrying in {wait:.0f}s")
                time.sleep(wait)

        raise RuntimeError(f"_read_band_window failed after {self.max_retries} attempts") from last_exc

    # multi-tile mosaicking per band

    def _fetch_mosaicked_band(
        self, band_name: str, items: List[Any], grid_info: Dict
    ) -> Tuple[str, Optional[np.ndarray]]:
        """reads and seamlessly overlays all tiles captured on the same date onto master canvas."""
        is_scl = (band_name == "SCL")
        resampling = Resampling.nearest if is_scl else Resampling.bilinear
        dst_nodata = 0.0 if is_scl else np.nan

        # target canvas
        canvas_shape = grid_info["shape"]
        combined_canvas = np.zeros(canvas_shape, dtype=np.uint8) if is_scl else np.full(canvas_shape, np.nan, dtype=np.float32)

        for item in items:
            if band_name not in item.assets:
                continue

            try:
                asset_url = item.assets[band_name].href
                raw_data, src_transform, src_crs = self._read_band_window(
                    asset_url, grid_info["bbox_wgs84"]
                )

                aligned = self.aligner.align_raster_to_master(
                    src_array=raw_data,
                    src_crs=src_crs,
                    src_transform=src_transform,
                    grid_info=grid_info,
                    resampling_method=resampling,
                    dst_nodata=dst_nodata,
                )

                if is_scl:
                    # overlay non-zero classifications
                    valid_mask = (aligned != 0)
                    combined_canvas[valid_mask] = aligned[valid_mask].astype(np.uint8)
                else:
                    # overlay non-nan and positive reflectance pixels
                    valid_mask = np.isfinite(aligned) & (aligned > 0.0)
                    combined_canvas[valid_mask] = aligned[valid_mask]

            except Exception as e:
                print(f"  [Warning] failed reading tile {item.id} for band {band_name}: {e}")
                continue

        return band_name, combined_canvas

    # sentinel-2 scene search with mosaicking

    def fetch_sentinel2_scene(
        self,
        grid_info: Dict,
        target_date_str: str,
        lookback_days: int = 25,
        max_cloud_cover: float = 60.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
        """searches and mosaics multi-tile acquisitions grouped by date."""
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
            print(f"  [S2] STAC search failed ({e}).")
            return {}, {}, ""

        if not items:
            return {}, {}, ""

        # group candidate tiles by acquisition date
        items_by_date: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            d_str = item.datetime.strftime("%Y-%m-%d")
            items_by_date[d_str].append(item)

        # evaluate candidate dates from newest to oldest
        sorted_dates = sorted(items_by_date.keys(), reverse=True)

        for candidate_date in sorted_dates:
            date_items = items_by_date[candidate_date]

            # 1. build mosaicked scl mask to test spatial coverage
            _, mosaicked_scl = self._fetch_mosaicked_band("SCL", date_items, grid_info)
            if mosaicked_scl is None:
                continue

            nodata_ratio = float((mosaicked_scl == 0).mean())
            if nodata_ratio > self.max_nodata_fraction:
                print(
                    f"  [Skip] date {candidate_date} ({len(date_items)} tiles) rejected: "
                    f"NoData {nodata_ratio * 100:.1f}% (> {self.max_nodata_fraction * 100:.0f}%)"
                )
                continue

            print(f"  [S2] selected date {candidate_date} mosaicking {len(date_items)} tile(s) (NoData: {nodata_ratio * 100:.1f}%)")

            # 2. fetch and mosaic all remaining bands in parallel
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
                        band = future_map[future]
                        print(f"  [Warning] thread exception for band {band}: {e}")

            masks = self.extract_scl_masks(mosaicked_scl)
            return aligned_bands, masks, candidate_date

        print(f"  [Error] no date found with combined NoData < {self.max_nodata_fraction * 100:.0f}%.")
        return {}, {}, ""

    # sentinel-1 sar fetch with multi-slice mosaicking

    def fetch_sentinel1_sar(
        self, grid_info: Dict, target_date_str: str, window_days: int = 12
    ) -> Dict[str, np.ndarray]:
        """fetches and mosaics sentinel-1 rtc sar (vv + vh) slices."""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=window_days)
        end_dt = target_dt + timedelta(days=window_days)

        try:
            search = self.stac_client.search(
                collections=["sentinel-1-rtc"],
                bbox=grid_info["bbox_wgs84"],
                datetime=f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}",
            )
            items = list(search.items())
        except Exception as e:
            print(f"  [S1] STAC search failed ({e}).")
            return {}

        if not items:
            return {}

        # group slices by date
        items_by_date: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            d_str = item.datetime.strftime("%Y-%m-%d")
            items_by_date[d_str].append(item)

        # find closest date to target
        sorted_dates = sorted(
            items_by_date.keys(),
            key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d") - target_dt).days),
        )

        for d_str in sorted_dates:
            date_items = items_by_date[d_str]
            sar_bands: Dict[str, np.ndarray] = {}

            for pol in ["vv", "vh"]:
                _, pol_canvas = self._fetch_mosaicked_band(pol, date_items, grid_info)
                if pol_canvas is not None and not np.isnan(pol_canvas).all():
                    sar_bands[f"SAR_{pol.upper()}"] = pol_canvas

            if len(sar_bands) == 2:
                return sar_bands

        print("  [S1] no complete mosaicked VV+VH pair found in search window.")
        return {}

    # combined entry point

    def fetch_all_radar_optical(self, lat: float, lon: float, target_date: str) -> Dict[str, Any]:
        grid_info = self.aligner.get_master_grid_info(lat, lon)

        # 1. fetch t0 with mosaicking
        bands_t0, masks_t0, t0_date = self.fetch_sentinel2_scene(
            grid_info, target_date, lookback_days=15
        )
        if not bands_t0:
            return {}

        # 2. fetch tprev with mosaicking
        t0_dt = datetime.strptime(t0_date, "%Y-%m-%d")
        tprev_target = (t0_dt - timedelta(days=10)).strftime("%Y-%m-%d")
        bands_tprev, masks_tprev, tprev_date = self.fetch_sentinel2_scene(
            grid_info, tprev_target, lookback_days=30
        )

        # 3. fetch s1 sar
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fetcher = SentinelFetcher()

    test_lat, test_lon = 62.10869, -140.44048
    test_date = "2021-06-15"

    print(f"fetching mosaicked sentinel data for [{test_lat}, {test_lon}] on {test_date}...")
    t_start = time.perf_counter()

    result = fetcher.fetch_all_radar_optical(lat=test_lat, lon=test_lon, target_date=test_date)

    if not result:
        print("failed to find clear scenes in the specified range.")
    else:
        elapsed = time.perf_counter() - t_start
        print(f"\nsuccessfully fetched in {elapsed:.2f}s")
        print(f"t0 date: {result['t0_date']}")
        print(f"tprev date: {result['tprev_date']}")

        layers_to_show = []

        for band, arr in result["bands_t0"].items():
            layers_to_show.append((f"T0 Optical: {band} ({result['t0_date']})", arr, "viridis"))

        for mask_name, arr in result["masks_t0"].items():
            layers_to_show.append((f"T0 Mask: {mask_name}", arr, "gray"))

        for band, arr in result["bands_tprev"].items():
            layers_to_show.append((f"Tprev Optical: {band} ({result['tprev_date']})", arr, "magma"))

        for band, arr in result["sar_bands"].items():
            layers_to_show.append((f"SAR: {band} (near {result['t0_date']})", arr, "plasma"))

        print(f"\nvisualizing {len(layers_to_show)} layers sequentially...")
        print("close current window to view the next layer.\n")

        for title, arr, cmap in layers_to_show:
            fig, ax = plt.subplots(figsize=(8, 8))

            disp_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            valid_pixels = arr[np.isfinite(arr)]
            if len(valid_pixels) > 0 and "MASK" not in title and "SCL" not in title:
                vmin, vmax = np.percentile(valid_pixels, [2, 98])
            else:
                vmin, vmax = disp_arr.min(), disp_arr.max()

            interpolation = "nearest" if "MASK" in title or "SCL" in title else "bilinear"

            im = ax.imshow(
                disp_arr,
                cmap=cmap,
                interpolation=interpolation,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            ax.axis("off")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("pixel value")

            plt.tight_layout()
            plt.show()
            plt.close(fig)

        print("visualization completed.")