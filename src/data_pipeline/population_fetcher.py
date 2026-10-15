import os
from typing import Dict, Optional
import ee
import numpy as np
import requests
from dotenv import load_dotenv


class PopulationFetcher:
    """fetches aggregated population metrics (1D context) for local (10km) and regional (50km) radii."""

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
                f"failed to initialize earth engine: {e}\n"
                f"run 'ee.Authenticate()' or set GEE_PROJECT in .env"
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
        """calculates total population in 10km (local) and 50km (regional catchment) radii."""
        pop_image = self._get_population_image(year=year).unmask(0)
        point = ee.Geometry.Point([lon, lat])

        # local population (10 km radius covering the scene itself)
        buf_10k = point.buffer(10000)
        # regional population (50 km catchment radius for recreation/mobility pressure)
        buf_50k = point.buffer(50000)

        # we use reduceRegions to batch multiple radius sums in one GEE call
        fc = ee.FeatureCollection([
            ee.Feature(buf_10k, {'id': 'local_10k'}),
            ee.Feature(buf_50k, {'id': 'regional_50k'})
        ])

        try:
            results = pop_image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.sum(),
                scale=100
            ).getInfo()
            
            pop_10k = 0.0
            pop_50k = 0.0
            
            for feat in results.get('features', []):
                val = feat['properties'].get('sum', 0.0)
                if feat['properties'].get('id') == 'local_10k':
                    pop_10k = float(val)
                else:
                    pop_50k = float(val)

            # return log-scaled values to prevent massive urban centers from blowing up gradients
            return {
                "population_10km_log1p": float(np.log1p(pop_10k)),
                "population_50km_log1p": float(np.log1p(pop_50k)),
            }
            
        except Exception as e:
            print(f"[PopulationFetcher] 1D GEE fetch failed: {e}")
            return {
                "population_10km_log1p": 0.0,
                "population_50km_log1p": 0.0,
            }