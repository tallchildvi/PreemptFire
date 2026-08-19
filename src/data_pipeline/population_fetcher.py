import io
import os
from typing import Optional

import ee
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject

from src.processing.grid_aligner import GridAligner

from dotenv import load_dotenv

class PopulationFetcher:
    """
    Fetches genuine 100m WorldPop demographic data via Google Earth Engine API (GEE),
    streams the target bbox into RAM (< 100 KB), reprojects to the 10m master UTM grid
    via nearest resampling, and applies log1p compression. 0 MB disk footprint.
    """

    def __init__(self, project_id: Optional[str] = None):
        load_dotenv()
        self.aligner = GridAligner()
        self.project_id = os.getenv("GEE_PROJECT")
        self._init_earth_engine()

        # Primary collection (2015-2030) and official fallback (2000-2021)
        self.collection_primary = "projects/sat-io/open-datasets/WORLDPOP/pop"
        self.collection_fallback = "WorldPop/GP/100m/pop"

    def _init_earth_engine(self):
        """Initializes GEE client with credentials or GCP project ID."""
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Earth Engine: {e}\n"
                f"Run 'ee.Authenticate()' in your terminal or provide a valid GEE_PROJECT ID."
            )

    def _get_population_image(self, year: int) -> ee.Image:
        """
        Filters WorldPop image collection for the specific year with fallback logic.
        """
        # 1. Attempt to fetch from catalog
        try:
            col = ee.ImageCollection(self.collection_primary)
            filtered = col.filter(ee.Filter.calendarRange(year, year, "year"))
            if filtered.size().getInfo() > 0:
                return filtered.select(["population"]).mosaic()
        except Exception:
            pass

        # 2. Fallback to official WorldPop/GP/100m/pop collection
        col_official = ee.ImageCollection(self.collection_fallback)
        filtered_official = col_official.filter(ee.Filter.calendarRange(year, year, "year"))
        
        if filtered_official.size().getInfo() > 0:
            return filtered_official.select(["population"]).mosaic()

        # 3. If target year is out of range, take the latest available slice
        return col_official.select(["population"]).sort("system:time_start", False).first()

    def fetch_population(
        self,
        grid_info: dict,
        year: int = 2020
    ) -> np.ndarray:
        """
        Streams 100m WorldPop data for the scene bbox from GEE into memory,
        reprojects onto the 10m master UTM grid (2048x2048), and applies log1p.

        Returns:
            np.ndarray: (2048, 2048) float32 array with log1p population count.
        """
        shape = grid_info["shape"]          # (2048, 2048)
        master_crs = grid_info["crs"]
        master_transform = grid_info["transform"]
        min_lon, min_lat, max_lon, max_lat = grid_info["bbox_wgs84"]

        # 1. Build BBOX geometry for GEE query
        # Add small spatial buffer (~0.01 deg) to avoid boundary null pixels during reprojection
        buf = 0.01
        region = ee.Geometry.Rectangle(
            [min_lon - buf, min_lat - buf, max_lon + buf, max_lat + buf],
            proj="EPSG:4326",
            geodesic=False
        )

        pop_image = self._get_population_image(year=year)

        # 2. Get direct GeoTIFF stream URL via getDownloadURL
        try:
            download_url = pop_image.getDownloadURL({
                "region": region,
                "scale": 100,          # Native WorldPop resolution
                "crs": "EPSG:4326",
                "format": "GEO_TIFF"
            })

            resp = requests.get(download_url, timeout=30)
            resp.raise_for_status()

        except Exception as e:
            print(f"[PopulationFetcher] GEE download failed: {e}")
            return np.zeros(shape, dtype=np.float32)

        # 3. Read GeoTIFF from memory (0 bytes on disk) and reproject to Master Grid
        pop_10m = np.zeros(shape, dtype=np.float32)

        with MemoryFile(resp.content) as memfile:
            with memfile.open() as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=pop_10m,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=master_crs,
                    dst_transform=master_transform,
                    resampling=Resampling.nearest,  # Preserves discrete demographic count blocks
                    src_nodata=src.nodata,
                    dst_nodata=0.0
                )

        # 4. Clean NoData and negative values
        pop_10m = np.nan_to_num(pop_10m, nan=0.0)
        pop_10m = np.clip(pop_10m, 0.0, None)

        # 5. Log1p compression: log(1 + pop)
        return np.log1p(pop_10m).astype(np.float32)


if __name__ == "__main__":
    aligner = GridAligner()
    # Test scene in downtown Edmonton
    grid_info = aligner.get_master_grid_info(lat=53.5461, lon=-113.4937)

    # Uses GEE_PROJECT environment variable or default credentials:
    fetcher = PopulationFetcher()
    print(fetcher.project_id)
    pop_layer = fetcher.fetch_population(grid_info=grid_info, year=2023)

    print("\n--- GEE WorldPop Population Results ---")
    print(f"Shape:      {pop_layer.shape}")
    print(f"Dtype:      {pop_layer.dtype}")
    print(f"Min:        {pop_layer.min():.4f}")
    print(f"Max:        {pop_layer.max():.4f} (log1p)")
    print(f"Mean:       {pop_layer.mean():.4f}")
    print(f"Non-zero %: {(pop_layer > 0).mean() * 100:.2f}%")