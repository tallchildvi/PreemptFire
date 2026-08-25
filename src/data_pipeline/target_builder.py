from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine, from_origin


class TargetBuilder:
    """Continuous multi-scale ground-truth target generator for wildfire ignition risk.

    Generates smooth, unclipped 4-channel Gaussian fields across the entire raster grid:
        Channel 0: Immediate Footprint   (sigma = 250m)
        Channel 1: Sub-Regional Hazard   (sigma = 1000m)
        Channel 2: Macro Risk Field      (sigma = 3000m)
        Channel 3: Regional Macro-Field  (sigma = 4000m)
    """

    SIGMAS_M: Tuple[float, float, float, float] = (250.0, 1000.0, 3000.0, 4000.0)

    def __init__(self, pixel_size_m: float = 10.0):
        self.pixel_size_m = float(pixel_size_m)

    def build_target(
        self,
        grid_info: Dict,
        lat: float,
        lon: float,
        is_fire: int | bool,
    ) -> np.ndarray:
        """Generates continuous 4-channel target tensor without artificial cutoffs.

        If is_fire == 0 / False -> returns zeros array.
        If is_fire == 1 / True  -> calculates exact full-grid Gaussians.
        """
        h, w = grid_info["shape"]
        num_channels = len(self.SIGMAS_M)
        target = np.zeros((num_channels, h, w), dtype=np.float32)

        if not is_fire:
            return target
        
        transformer = Transformer.from_crs(
            "EPSG:4326", grid_info["crs"], always_xy=True
        )
        x_utm, y_utm = transformer.transform(lon, lat)

        inv_transform: Affine = ~grid_info["transform"]
        col_f, row_f = inv_transform * (x_utm, y_utm)

        if not (0 <= row_f < h and 0 <= col_f < w):
            return target

        y, x = np.ogrid[:h, :w]
        dist_sq_m = ((y - row_f) ** 2 + (x - col_f) ** 2) * (self.pixel_size_m ** 2)

        for ch_idx, sigma_m in enumerate(self.SIGMAS_M):
            target[ch_idx] = np.exp(-dist_sq_m / (2.0 * (sigma_m ** 2))).astype(np.float32)

        return target


if __name__ == "__main__":
    grid_size = 1024
    pixel_size = 10.0
    center_lat, center_lon = 53.5461, -113.4937

    transformer_fwd = Transformer.from_crs(
        "EPSG:4326", "EPSG:32612", always_xy=True
    )
    center_x, center_y = transformer_fwd.transform(center_lon, center_lat)

    origin_x = center_x - (grid_size / 2) * pixel_size
    origin_y = center_y + (grid_size / 2) * pixel_size

    mock_grid_info = {
        "shape": (grid_size, grid_size),
        "crs": "EPSG:32612",
        "transform": from_origin(origin_x, origin_y, pixel_size, pixel_size),
    }

    builder = TargetBuilder(pixel_size_m=pixel_size)
    target_tensor = builder.build_target(
        grid_info=mock_grid_info,
        lat=center_lat,
        lon=center_lon,
        is_fire=1,
    )

    scales_metadata = [
        ("Immediate Footprint", "inferno", 250.0),
        ("Sub-Regional Hazard", "magma", 1000.0),
        ("Macro Risk Field", "viridis", 3000.0),
        ("Regional Macro-Field", "plasma", 4000.0),
    ]

    print(f"Generated target tensor shape: {target_tensor.shape}")
    print("Sequential rendering: close each window to view the next continuous scale...\n")

    center_px = grid_size // 2

    for idx, (title, cmap, sigma_m) in enumerate(scales_metadata, start=1):
        dome = target_tensor[idx - 1]
        min_val = dome.min()
        max_val = dome.max()

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(dome, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)

        ax.plot(
            center_px,
            center_px,
            marker="x",
            color="red",
            markersize=10,
            label="Ignition Center",
        )

        ax.set_title(
            f"Step {idx}/4: {title}\n"
            f"sigma = {sigma_m:.0f} m | Min Value: {min_val:.4f} | Max Value: {max_val:.4f}",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("X (pixels, 1 px = 10 m)", fontsize=10)
        ax.set_ylabel("Y (pixels, 1 px = 10 m)", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.85)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Continuous Target Risk", fontsize=10)

        plt.tight_layout()
        plt.show()
        plt.close(fig)
