from typing import Any, Dict, Generator, List
import matplotlib.pyplot as plt
import numpy as np

from src.config import MAX_INVALID_RATIO, PATCH_SIZE_PX, STRIDE_PX


class PatchExtractor:
    """Extracts and validates raw full-scene patches (128x128 @ 100m) without precomputed indices or targets."""

    # Fixed contract of 22 pure raw 2D channels
    CANONICAL_2D_CHANNELS: List[str] = [
        # Sentinel-2 L2A Optical bands (T0) - 10 channels
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
        # Sentinel-1 RTC SAR backscatter - 2 channels
        "SAR_VV", "SAR_VH",
        # Copernicus DEM topographic features - 4 channels
        "Elevation", "Slope", "Northness", "Eastness",
        # Historical OSM accessibility and proximity - 5 channels
        "Travel_Time_Roads", "Travel_Time_Trails",
        "Dist_to_Railways", "Dist_to_Camps", "Dist_to_Powerlines",
        # ERA5-Land volumetric surface soil moisture - 1 channel
        "Soil_Moisture",
    ]

    def __init__(
        self,
        patch_size: int = PATCH_SIZE_PX,
        stride: int = STRIDE_PX,
        max_invalid_ratio: float = MAX_INVALID_RATIO,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.max_invalid_ratio = max_invalid_ratio

    def extract_patches(
        self,
        sample: Dict[str, Any],
        scene_id: str,
    ) -> Generator[Dict[str, Any], None, None]:
        rasters_2d = sample["rasters_2d"]
        loss_mask = sample["loss_mask"]
        scl = sample["scl"]
        context_1d = sample["context_1d"]
        metadata = sample["metadata"]

        h_full, w_full = loss_mask.shape

        sar_vv = rasters_2d.get("SAR_VV")
        if sar_vv is not None and isinstance(sar_vv, np.ndarray):
            sar_valid_mask = (np.isfinite(sar_vv) & (sar_vv > 0.0)).astype(np.float32)
        else:
            sar_valid_mask = np.zeros((h_full, w_full), dtype=np.float32)

        canonical_rasters = []
        for ch in self.CANONICAL_2D_CHANNELS:
            if ch in ("SAR_VV", "SAR_VH"):
                arr = rasters_2d.get(ch)
                if arr is not None:
                    canonical_rasters.append(
                        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    )
                else:
                    canonical_rasters.append(np.zeros((h_full, w_full), dtype=np.float32))
            elif ch in rasters_2d and rasters_2d[ch] is not None:
                canonical_rasters.append(
                    np.nan_to_num(rasters_2d[ch], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                )
            else:
                raise KeyError(f"Mandatory raw 2D channel '{ch}' is missing from collected sample.")

        stacked_2d = np.stack(canonical_rasters, axis=0).astype(np.float32)

        sorted_context_keys = sorted(context_1d.keys())
        context_vec = np.array([float(context_1d[k]) for k in sorted_context_keys], dtype=np.float32)

        patch_idx = 0

        for r in range(0, h_full - self.patch_size + 1, self.stride):
            for c in range(0, w_full - self.patch_size + 1, self.stride):
                row_slice = slice(r, r + self.patch_size)
                col_slice = slice(c, c + self.patch_size)

                patch_loss_mask = loss_mask[row_slice, col_slice]
                invalid_ratio = float((patch_loss_mask > 0.5).mean())
                if invalid_ratio > self.max_invalid_ratio:
                    continue

                patch_2d = stacked_2d[:, row_slice, col_slice]
                patch_sar_mask = sar_valid_mask[row_slice, col_slice]
                patch_scl = scl[row_slice, col_slice]
                patch_has_sar = float(patch_sar_mask.mean() >= 0.50)

                validity_flags = [
                    patch_has_sar if ch in ("SAR_VV", "SAR_VH") else 1.0
                    for ch in self.CANONICAL_2D_CHANNELS
                ]
                patch_channel_validity = np.array(validity_flags, dtype=np.float32)

                patch_meta = {
                    "scene_id": str(scene_id),
                    "patch_idx": int(patch_idx),
                    "row_offset": int(r),
                    "col_offset": int(c),
                    "center_row": float(metadata.get("center_row", self.patch_size / 2.0)) - r,
                    "center_col": float(metadata.get("center_col", self.patch_size / 2.0)) - c,
                    "invalid_ratio": float(invalid_ratio),
                    "sar_coverage_ratio": float(patch_sar_mask.mean()),
                    "has_sar": int(patch_has_sar > 0.5),
                    "lat": float(metadata["lat"]),
                    "lon": float(metadata["lon"]),
                    "target_date": str(metadata["target_date"]),
                    "t0_date": str(metadata["t0_date"]),
                    "crs": str(metadata["crs"]),
                    "utm_bounds": metadata.get("utm_bounds", []),
                    "is_fire": int(metadata["is_fire"]),
                }

                yield {
                    "patch_id": f"{scene_id}_p{patch_idx:03d}",
                    "X_2d": patch_2d,
                    "X_1d": context_vec,
                    "channel_validity": patch_channel_validity,
                    "scl": patch_scl,
                    "is_fire": int(metadata["is_fire"]),
                    "metadata": patch_meta,
                    "channel_names_2d": self.CANONICAL_2D_CHANNELS,
                    "context_names_1d": sorted_context_keys,
                }

                patch_idx += 1


if __name__ == "__main__":
    from src.config import GRID_SIZE_PX

    extractor = PatchExtractor()
    shape = (GRID_SIZE_PX, GRID_SIZE_PX)

    mock_rasters = {
        ch: np.random.uniform(0.1, 1.0, size=shape).astype(np.float32)
        for ch in PatchExtractor.CANONICAL_2D_CHANNELS
    }
    mock_sample = {
        "rasters_2d": mock_rasters,
        "loss_mask": np.zeros(shape, dtype=np.float32),
        "scl": np.full(shape, 4, dtype=np.uint8),
        "context_1d": {"target_temperature_max": 32.5, "days_since_rain": 14.0},
        "metadata": {
            "lat": 41.6986,
            "lon": -7.0011,
            "center_row": 64.0,
            "center_col": 64.0,
            "target_date": "2022-08-09",
            "t0_date": "2022-08-05",
            "crs": "EPSG:32629",
            "utm_bounds": [0, 0, 12800, 12800],
            "is_fire": 1,
        },
    }

    patches = list(extractor.extract_patches(mock_sample, scene_id="TEST_RAW_SCENE"))
    print(f"Extracted {len(patches)} raw patch(es) | X_2d shape: {patches[0]['X_2d'].shape} | is_fire: {patches[0]['is_fire']}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(patches[0]["X_2d"][0], cmap="viridis")
    ax.set_title(f"Raw Full-Scene Patch (B02)\nShape: {patches[0]['X_2d'].shape}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()