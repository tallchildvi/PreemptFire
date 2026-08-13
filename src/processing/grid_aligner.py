import math
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from pyproj import Transformer

from src.config import GRID_SIZE_PX, HALF_SCENE_METERS, PIXEL_SCALE_METERS


class GridAligner:
    """calculates utm master grid metadata and resamples rasters to 10m 2048x2048 grid"""

    @staticmethod
    def get_utm_epsg(lat: float, lon: float) -> int:
        """calculate utm epsg code based on latitude and longitude"""
        zone = math.floor((lon + 180) / 6) + 1
        return (32600 + zone) if lat >= 0 else (32700 + zone)

    def get_master_grid_info(self, lat: float, lon: float) -> dict:
        """generates spatial reference info for a 20.48x20.48 km scene centered at (lat, lon)"""
        epsg_code = self.get_utm_epsg(lat, lon)
        utm_crs = CRS.from_epsg(epsg_code)

        transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        center_x_utm, center_y_utm = transformer_to_utm.transform(lon, lat)

        min_x = center_x_utm - HALF_SCENE_METERS
        max_x = center_x_utm + HALF_SCENE_METERS
        min_y = center_y_utm - HALF_SCENE_METERS
        max_y = center_y_utm + HALF_SCENE_METERS

        transform = from_bounds(min_x, min_y, max_x, max_y, GRID_SIZE_PX, GRID_SIZE_PX)

        # precise wgs84 bounding box calculation using all 4 utm corners
        transformer_to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        corners_utm = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y)
        ]
        corners_wgs84 = [transformer_to_wgs84.transform(x, y) for x, y in corners_utm]

        lons = [pt[0] for pt in corners_wgs84]
        lats = [pt[1] for pt in corners_wgs84]

        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)

        return {
            "center_lat_lon": (lat, lon),
            "epsg": epsg_code,
            "crs": utm_crs,
            "transform": transform,
            "shape": (GRID_SIZE_PX, GRID_SIZE_PX),
            "utm_bounds": (min_x, min_y, max_x, max_y),
            "bbox_wgs84": [round(west, 5), round(south, 5), round(east, 5), round(north, 5)]
        }

    def align_raster_to_master(
        self,
        src_array: np.ndarray,
        src_crs: CRS,
        src_transform: rasterio.Affine,
        grid_info: dict,
        resampling_method: Resampling,
        src_nodata: float | int | None = None,
        dst_nodata: float | int = np.nan
    ) -> np.ndarray:
        """reprojects and resamples raster array to match master grid with explicit nodata handling"""
        
        # initialize array with nodata values instead of zeros
        if np.isnan(dst_nodata):
            dst_array = np.full(grid_info["shape"], np.nan, dtype=np.float32)
        else:
            dst_array = np.full(grid_info["shape"], dst_nodata, dtype=src_array.dtype)

        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=grid_info["transform"],
            dst_crs=grid_info["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling_method
        )

        return dst_array


if __name__ == "__main__":
    aligner = GridAligner()
    grid_info = aligner.get_master_grid_info(lat=53.5461, lon=-113.4937)

    print(f"utm zone epsg: {grid_info['epsg']}")
    print(f"wgs84 bbox for apis: {grid_info['bbox_wgs84']}")
    print(f"master scene shape: {grid_info['shape']}")