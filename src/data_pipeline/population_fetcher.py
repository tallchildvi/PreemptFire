import os
from typing import Dict, Optional
from dotenv import load_dotenv
import ee
import numpy as np


class PopulationFetcher:
    """Fetches aggregated population metrics (1D context) for local (10km) and regional (50km) radii."""

    def __init__(self, project_id: Optional[str] = None):
        load_dotenv()
        self.project_id = project_id or os.getenv("GEE_PROJECT")
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

    def fetch_population_1d(self, lat: float, lon: float, year: int = 2020) -> Dict[str, float]:
        """Calculates total log1p population in 10km (local) and 50km (regional catchment) radii."""
        pop_image = self._get_population_image(year=year).unmask(0)
        point = ee.Geometry.Point([lon, lat])

        buf_10k = point.buffer(10000)
        buf_50k = point.buffer(50000)

        fc = ee.FeatureCollection([
            ee.Feature(buf_10k, {"id": "local_10k"}),
            ee.Feature(buf_50k, {"id": "regional_50k"}),
        ])

        try:
            results = pop_image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.sum(),
                scale=100,
            ).getInfo()

            pop_10k = 0.0
            pop_50k = 0.0

            for feat in results.get("features", []):
                val = feat["properties"].get("sum", 0.0)
                if feat["properties"].get("id") == "local_10k":
                    pop_10k = max(0.0, float(val))
                else:
                    pop_50k = max(0.0, float(val))

            return {
                "population_10km_log1p": float(np.log1p(pop_10k)),
                "population_50km_log1p": float(np.log1p(pop_50k)),
            }

        except Exception as e:
            print(f"  [PopulationFetcher] 1D GEE fetch failed: {e}")
            return {
                "population_10km_log1p": 0.0,
                "population_50km_log1p": 0.0,
            }


if __name__ == "__main__":
    fetcher = PopulationFetcher()
    test_lat, test_lon = 41.6986, -7.0011

    print(f"Testing PopulationFetcher for ({test_lat}, {test_lon})...")
    pop_metrics = fetcher.fetch_population_1d(lat=test_lat, lon=test_lon, year=2020)
    for k, v in pop_metrics.items():
        raw_est = np.expm1(v)
        print(f"  {k:<24}: {v:.4f} (approx. {raw_est:,.0f} people)")