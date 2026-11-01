from datetime import datetime, timedelta
import os
from typing import Optional
from dotenv import load_dotenv
import ee
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject
import requests

from src.processing.grid_aligner import GridAligner


class ERA5Fetcher:
    """Fetches daily volumetric surface soil moisture (0-7 cm) from ECMWF ERA5-Land via GEE."""

    def __init__(self, project_id: Optional[str] = None, default_moisture: float = 0.20):
        load_dotenv()
        self.project_id = project_id or os.getenv("GEE_PROJECT")
        self.default_moisture = float(default_moisture)
        self.aligner = GridAligner()
        self._init_earth_engine()
        self.collection_name = "ECMWF/ERA5_LAND/DAILY_AGGR"

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

    def fetch_soil_moisture(self, grid_info: dict, target_date: str) -> np.ndarray:
        """Fetches top-layer (0-7 cm) volumetric soil moisture aligned to the master grid."""
        shape = grid_info["shape"]
        master_crs = grid_info["crs"]
        master_transform = grid_info["transform"]
        min_lon, min_lat, max_lon, max_lat = grid_info["bbox_wgs84"]

        buf = 0.15
        region = ee.Geometry.Rectangle(
            [min_lon - buf, min_lat - buf, max_lon + buf, max_lat + buf],
            proj="EPSG:4326",
            geodesic=False,
        )

        target_dt = datetime.strptime(target_date.split(" ")[0], "%Y-%m-%d")
        start_str = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end_str = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        col = (
            ee.ImageCollection(self.collection_name)
            .filterDate(start_str, end_str)
            .select(["volumetric_soil_water_layer_1"])
        )

        try:
            img = col.sort("system:time_start", False).first().unmask(self.default_moisture)
            download_url = img.getDownloadURL({
                "region": region,
                "scale": 5000,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            })
            resp = requests.get(download_url, timeout=30)
            resp.raise_for_status()

            soil_raster = np.full(shape, self.default_moisture, dtype=np.float32)
            with MemoryFile(resp.content) as memfile:
                with memfile.open() as src:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=soil_raster,
                        src_crs=src.crs,
                        src_transform=src.transform,
                        dst_crs=master_crs,
                        dst_transform=master_transform,
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=self.default_moisture,
                    )

            soil_raster = np.nan_to_num(soil_raster, nan=self.default_moisture)
            return np.clip(soil_raster, 0.0, 1.0).astype(np.float32)

        except Exception as e:
            print(f"  [ERA5Fetcher] Warning: soil moisture fetch failed ({e}), using fallback={self.default_moisture}")
            return np.full(shape, self.default_moisture, dtype=np.float32)


if __name__ == "__main__":
    aligner = GridAligner()
    fetcher = ERA5Fetcher()

    test_lat, test_lon = 41.6986, -7.0011
    test_date = "2022-08-08"
    grid_info = aligner.get_master_grid_info(test_lat, test_lon)

    print(f"Fetching ERA5-Land Soil Moisture for [{test_lat}, {test_lon}] on {test_date}...")
    sm = fetcher.fetch_soil_moisture(grid_info=grid_info, target_date=test_date)
    print(f"Shape: {sm.shape} | Range: [{sm.min():.4f}, {sm.max():.4f}] | Mean: {sm.mean():.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sm, cmap="YlGnBu", origin="upper", vmin=0.0, vmax=0.6)
    ax.set_title(
        f"ERA5-Land Volumetric Soil Moisture (0-7 cm)\nDate: {test_date} | Mean: {sm.mean():.3f} m3/m3",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Soil Moisture (m3/m3)")
    plt.tight_layout()
    plt.show()