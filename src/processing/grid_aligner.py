import math
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject

from src.config import GRID_SIZE_PX, HALF_SCENE_METERS, PIXEL_SCALE_METERS


class GridAligner:
    """Calculates UTM master grid metadata and resamples rasters to the configured 100m master grid."""

    @staticmethod
    def get_utm_epsg(lat: float, lon: float) -> int:
        """Calculates UTM EPSG code based on latitude and longitude."""
        zone = math.floor((lon + 180.0) / 6.0) + 1
        return (32600 + zone) if lat >= 0 else (32700 + zone)

    def get_master_grid_info(self, lat: float, lon: float) -> dict:
        """Generates spatial reference metadata for a scene centered at (lat, lon)."""
        epsg_code = self.get_utm_epsg(lat, lon)
        utm_crs = CRS.from_epsg(epsg_code)

        transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        center_x_utm, center_y_utm = transformer_to_utm.transform(lon, lat)

        min_x = center_x_utm - HALF_SCENE_METERS
        max_x = center_x_utm + HALF_SCENE_METERS
        min_y = center_y_utm - HALF_SCENE_METERS
        max_y = center_y_utm + HALF_SCENE_METERS

        transform = from_bounds(min_x, min_y, max_x, max_y, GRID_SIZE_PX, GRID_SIZE_PX)

        transformer_to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        corners_utm = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
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
            "pixel_scale_m": PIXEL_SCALE_METERS,
            "utm_bounds": (min_x, min_y, max_x, max_y),
            "bbox_wgs84": [round(west, 5), round(south, 5), round(east, 5), round(north, 5)],
        }

    @staticmethod
    def latlon_to_pixel(lat: float, lon: float, grid_info: dict) -> Tuple[float, float]:
        """Converts (lat, lon) into continuous (row, col) coordinates within the master grid."""
        transformer = Transformer.from_crs("EPSG:4326", grid_info["crs"], always_xy=True)
        x_utm, y_utm = transformer.transform(lon, lat)
        inv_transform: Affine = ~grid_info["transform"]
        col_f, row_f = inv_transform * (x_utm, y_utm)
        return float(row_f), float(col_f)

    def align_raster_to_master(
        self,
        src_array: np.ndarray,
        src_crs: CRS | str,
        src_transform: rasterio.Affine,
        grid_info: dict,
        resampling_method: Resampling = Resampling.average,
        src_nodata: float | int | None = None,
        dst_nodata: float | int = np.nan,
    ) -> np.ndarray:
        """Reprojects and resamples a source raster array onto the master grid."""
        if src_crs is None or src_transform is None:
            raise ValueError("Source CRS and Affine transform must be provided for alignment.")

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
            resampling=resampling_method,
        )

        return dst_array


if __name__ == "__main__":
    aligner = GridAligner()
    test_lat, test_lon = 41.6986, -7.0011
    info = aligner.get_master_grid_info(lat=test_lat, lon=test_lon)
    row_f, col_f = aligner.latlon_to_pixel(test_lat, test_lon, info)

    print("GridAligner verification:")
    print(f"Center (lat, lon) : {info['center_lat_lon']} -> pixel (row={row_f:.2f}, col={col_f:.2f})")
    print(f"UTM Zone EPSG     : {info['epsg']}")
    print(f"Master Grid Shape : {info['shape']} @ {info['pixel_scale_m']}m/px")
    print(f"WGS84 BBox        : {info['bbox_wgs84']}")

    h, w = info["shape"]
    y, x = np.ogrid[:h, :w]
    dist_km = np.sqrt((y - row_f) ** 2 + (x - col_f) ** 2) * (PIXEL_SCALE_METERS / 1000.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_km, cmap="viridis", origin="upper")
    ax.plot(col_f, row_f, marker="x", color="red", markersize=10, label="Scene Center")
    ax.set_title(
        f"Master Grid Distance Field\nShape: {h}x{w} px ({PIXEL_SCALE_METERS:.0f}m/px) | EPSG:{info['epsg']}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Distance from Center (km)")
    plt.tight_layout()
    plt.show()