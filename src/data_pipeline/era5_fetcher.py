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

from src.processing.grid_aligner import GridAligner


class ERA5Fetcher:
    """
    Fetches volumetric soil moisture from ECMWF ERA5-Land Hourly via GEE,
    aggregates to daily mean, and reprojects onto the 10m master UTM grid (2048x2048).
    """

    COLLECTION_HOURLY = "ECMWF/ERA5_LAND/HOURLY"

    def __init__(self, project_id: Optional[str] = None, buffer_meters: float = 20_000.0):
        load_dotenv()
        self.aligner = GridAligner()
        self.project_id = project_id or os.getenv("GEE_PROJECT")
        self.buffer_meters = buffer_meters
        self._init_earth_engine()

    def _init_earth_engine(self) -> None:
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

    def _get_buffered_region(self, grid_info: dict) -> ee.Geometry:
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        b_min_x, b_max_x = min_x - self.buffer_meters, max_x + self.buffer_meters
        b_min_y, b_max_y = min_y - self.buffer_meters, max_y + self.buffer_meters

        transformer = Transformer.from_crs(grid_info["crs"], "EPSG:4326", always_xy=True)
        corners = [(b_min_x, b_min_y), (b_min_x, b_max_y), (b_max_x, b_min_y), (b_max_x, b_max_y)]
        coords_wgs84 = [transformer.transform(x, y) for x, y in corners]
        lons = [p[0] for p in coords_wgs84]
        lats = [p[1] for p in coords_wgs84]

        return ee.Geometry.Rectangle(
            [min(lons), min(lats), max(lons), max(lats)],
            proj="EPSG:4326",
            geodesic=False,
        )

    def fetch_soil_moisture(self, grid_info: dict, target_date: str) -> np.ndarray:
        """
        Streams volumetric soil moisture (0-7 cm, m3/m3) for target_date (YYYY-MM-DD)
        and reprojects onto 2048x2048 10m master grid.
        """
        shape = grid_info["shape"]
        region = self._get_buffered_region(grid_info)

        t_start = ee.Date(target_date)
        t_end = t_start.advance(1, "day")

        col = (
            ee.ImageCollection(self.COLLECTION_HOURLY)
            .filterDate(t_start, t_end)
            .select(["volumetric_soil_water_layer_1"])
        )

        if col.size().getInfo() == 0:
            print(f"[ERA5Fetcher] No ERA5 data found for {target_date}, trying 5-day fallback...")
            col = (
                ee.ImageCollection(self.COLLECTION_HOURLY)
                .filterDate(t_start.advance(-5, "day"), t_end)
                .select(["volumetric_soil_water_layer_1"])
            )
            if col.size().getInfo() == 0:
                return np.full(shape, 0.25, dtype=np.float32)

        daily_soil_img = col.mean()

        try:
            download_url = daily_soil_img.getDownloadURL({
                "region": region,
                "scale": 9000,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
            })
            resp = requests.get(download_url, timeout=40)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERA5Fetcher] GEE stream error: {e}")
            return np.full(shape, 0.25, dtype=np.float32)

        soil_10m = np.zeros(shape, dtype=np.float32)
        with MemoryFile(resp.content) as memfile:
            with memfile.open() as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=soil_10m,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=grid_info["crs"],
                    dst_transform=grid_info["transform"],
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=0.25,
                )

        soil_10m = np.nan_to_num(soil_10m, nan=0.25)
        return np.clip(soil_10m, 0.01, 0.75).astype(np.float32)


if __name__ == "__main__":
    aligner = GridAligner()

    lat, lon = 53.9057388, -113.2269477
    date = "2023-07-15"

    grid_info = aligner.get_master_grid_info(lat=lat, lon=lon)
    fetcher = ERA5Fetcher(buffer_meters=20_000.0)

    soil_moisture = fetcher.fetch_soil_moisture(grid_info=grid_info, target_date=date)

    print(f"\n--- ERA5-Land Soil Moisture (0-7 cm) ---")
    print(f"Shape: {soil_moisture.shape}")
    print(f"Dtype: {soil_moisture.dtype}")
    print(f"Range: [{soil_moisture.min():.3f}, {soil_moisture.max():.3f}] m3/m3")
    print(f"Mean:  {soil_moisture.mean():.3f} m3/m3")

    plt.figure(figsize=(8, 7))
    plt.imshow(soil_moisture, cmap="YlGnBu", origin="upper", vmin=0.1, vmax=0.45)
    plt.title(f"ERA5-Land Soil Moisture L1 (0-7 cm)\nDate: {date} | Coords: [{lat:.4f}, {lon:.4f}]", fontweight="bold")
    plt.axis("off")
    plt.colorbar(label="Volumetric Soil Water (m³/m³)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()