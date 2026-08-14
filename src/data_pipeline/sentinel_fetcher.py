from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from src.processing.grid_aligner import GridAligner


class SentinelFetcher:
    """fetches sentinel-2 (t0, tprev) and sentinel-1 sar bands from planetary computer stac"""

    def __init__(self):
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        self.aligner = GridAligner()

    def _read_band_window(self, asset_url: str, bbox_wgs84: list) -> tuple[np.ndarray, rasterio.Affine, any]:
        """reads a specific spatial window from a signed cloud cog url after projecting bounds to src crs"""
        with rasterio.open(asset_url) as src:
            west, south, east, north = bbox_wgs84
            
            # transform wgs84 degrees bounding box into raster native projection (meters)
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, west, south, east, north
            )
            
            window = from_bounds(left, bottom, right, top, src.transform)
            data = src.read(1, window=window, boundless=True, fill_value=np.nan)
            win_transform = src.window_transform(window)
            return data.astype(np.float32), win_transform, src.crs

    def fetch_sentinel2_scene(
        self, 
        grid_info: dict, 
        target_date_str: str, 
        lookback_days: int = 15,
        max_cloud_cover: float = 20.0
    ) -> tuple[dict, str]:
        """fetches clearest sentinel-2 l2a scene before target_date within bbox"""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)
        
        bbox = grid_info["bbox_wgs84"]
        date_range = f"{start_dt.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}"

        search = self.stac_client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            sortby=[{"field": "properties.datetime", "direction": "desc"}]
        )
        items = list(search.items())
        if not items:
            return {}, ""

        selected_item = items[0]
        actual_date = selected_item.datetime.strftime("%Y-%m-%d")

        bands_to_fetch = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"]
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
                dst_nodata=dst_nodata
            )

        return aligned_bands, actual_date

    def fetch_sentinel1_sar(
        self, 
        grid_info: dict, 
        target_date_str: str, 
        window_days: int = 10
    ) -> dict:
        """fetches sentinel-1 rtc sar (vv, vh polarizations) within date window"""
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=window_days)
        end_dt = target_dt + timedelta(days=window_days)

        bbox = grid_info["bbox_wgs84"]
        date_range = f"{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"

        search = self.stac_client.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=date_range
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
                dst_nodata=np.nan
            )

        return sar_bands

    def fetch_all_radar_optical(self, lat: float, lon: float, target_date: str) -> dict:
        """orchestrates fetching t0 optical, tprev optical, and s1 sar for a master scene"""
        grid_info = self.aligner.get_master_grid_info(lat, lon)
        
        # 1. fetch t0 optical channels
        bands_t0, t0_date = self.fetch_sentinel2_scene(grid_info, target_date, lookback_days=15)
        if not bands_t0:
            print(f"warning: no clear s2 t0 scene found for ({lat}, {lon}) near {target_date}")
            return {}

        # 2. fetch tprev optical channels (10 to 35 days before t0)
        t0_dt = datetime.strptime(t0_date, "%Y-%m-%d")
        tprev_target = (t0_dt - timedelta(days=10)).strftime("%Y-%m-%d")
        bands_tprev, tprev_date = self.fetch_sentinel2_scene(grid_info, tprev_target, lookback_days=25)

        # 3. fetch sentinel-1 sar (vv, vh)
        sar_bands = self.fetch_sentinel1_sar(grid_info, t0_date)

        return {
            "grid_info": grid_info,
            "t0_date": t0_date,
            "tprev_date": tprev_date,
            "bands_t0": bands_t0,
            "bands_tprev": bands_tprev,
            "sar_bands": sar_bands
        }

    def plot_preview_channels(self, result_dict: dict, downsample_step: int = 4):
        """downsamples and displays a grid preview of all collected channels and tci rgb"""
        if not result_dict:
            print("no data to visualize")
            return

        t0_bands = result_dict.get("bands_t0", {})
        tprev_bands = result_dict.get("bands_tprev", {})
        sar_bands = result_dict.get("sar_bands", {})

        plot_items = []

        # 1. generate downsampled rgb true color preview if channels exist
        if all(k in t0_bands for k in ["B04", "B03", "B02"]):
            r = t0_bands["B04"][::downsample_step, ::downsample_step]
            g = t0_bands["B03"][::downsample_step, ::downsample_step]
            b = t0_bands["B02"][::downsample_step, ::downsample_step]
            
            rgb = np.dstack([r, g, b])
            rgb = np.clip(rgb / 3000.0, 0.0, 1.0)
            rgb = np.nan_to_num(rgb, nan=0.0)
            plot_items.append(("T0 RGB (B04-B03-B02)", rgb, "rgb"))

        # 2. add t0 individual bands
        for name, matrix in t0_bands.items():
            downsampled = matrix[::downsample_step, ::downsample_step]
            cmap = "tab20" if name == "SCL" else "viridis"
            plot_items.append((f"T0: {name}", downsampled, cmap))

        # 3. add tprev bands
        for name, matrix in tprev_bands.items():
            downsampled = matrix[::downsample_step, ::downsample_step]
            cmap = "tab20" if name == "SCL" else "viridis"
            plot_items.append((f"Tprev: {name}", downsampled, cmap))

        # 4. add sar bands
        for name, matrix in sar_bands.items():
            downsampled = matrix[::downsample_step, ::downsample_step]
            plot_items.append((f"{name}", downsampled, "plasma"))

        total_plots = len(plot_items)
        cols = 4
        rows = math.ceil(total_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for idx, (title, img, cmap) in enumerate(plot_items):
            ax = axes[idx]
            if cmap == "rgb":
                ax.imshow(img)
            else:
                im = ax.imshow(img, cmap=cmap)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(title, fontsize=10)
            ax.axis("off")

        for idx in range(total_plots, len(axes)):
            axes[idx].axis("off")

        t0_d = result_dict.get("t0_date", "n/a")
        tp_d = result_dict.get("tprev_date", "n/a")
        fig.suptitle(f"scene preview (t0: {t0_d} | tprev: {tp_d})", fontsize=14, y=0.99)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    fetcher = SentinelFetcher()
    result = fetcher.fetch_all_radar_optical(lat=53.5461, lon=-113.4937, target_date="2023-06-15")

    if result:
        print(f"t0 date: {result['t0_date']}")
        print(f"tprev date: {result['tprev_date']}")
        fetcher.plot_preview_channels(result, downsample_step=4)