from typing import Any, Dict, Generator, List
import numpy as np

class PatchExtractor:
    """extracts and filters spatial patches from a master scene sample via generator streaming."""

    # 27 каналів: чіткий контракт із тренувальним ноутбуком
    CANONICAL_2D_CHANNELS: List[str] = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
        "B11", "B12", "SAR_VV", "SAR_VH", "Elevation", "Slope",
        "Northness", "Eastness", "Travel_Time_Roads", "Travel_Time_Trails",
        "Dist_to_Railways", "Dist_to_Camps", "Dist_to_Powerlines",
        "Soil_Moisture", "MASK_WATER", "MASK_SNOW", "MASK_CLOUDS",
        "MASK_CLOUD_SHADOWS", "MASK_SAR_VALID",
    ]

    def __init__(
        self,
        patch_size: int = 256,
        stride: int = 256,
        max_invalid_ratio: float = 0.20,
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
        target = sample["target"]
        context_1d = sample["context_1d"]
        metadata = sample["metadata"]

        h_full, w_full = loss_mask.shape

        sar_vv = rasters_2d.get("SAR_VV")
        if sar_vv is not None and isinstance(sar_vv, np.ndarray):
            sar_valid_mask = (np.isfinite(sar_vv) & (sar_vv > 0.0)).astype(np.float32)
            has_sar_scene = bool(sar_valid_mask.max() > 0.0)
        else:
            sar_valid_mask = np.zeros((h_full, w_full), dtype=np.float32)
            has_sar_scene = False

        canonical_rasters = []
        for ch in self.CANONICAL_2D_CHANNELS:
            if ch == "MASK_SAR_VALID":
                canonical_rasters.append(sar_valid_mask)
            elif ch in rasters_2d and rasters_2d[ch] is not None:
                canonical_rasters.append(
                    np.nan_to_num(rasters_2d[ch], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                )
            else:
                canonical_rasters.append(np.zeros((h_full, w_full), dtype=np.float32))

        stacked_2d = np.stack(canonical_rasters, axis=0).astype(np.float32)

        sorted_context_keys = sorted(context_1d.keys())
        context_vec = np.array([float(context_1d[k]) for k in sorted_context_keys], dtype=np.float32)

        patch_idx = 0

        for r in range(0, h_full - self.patch_size + 1, self.stride):
            for c in range(0, w_full - self.patch_size + 1, self.stride):
                row_slice = slice(r, r + self.patch_size)
                col_slice = slice(c, c + self.patch_size)

                patch_loss_mask = loss_mask[row_slice, col_slice]

                # Для режиму 100m loss_mask вже є часткою.
                invalid_ratio = float(patch_loss_mask.mean())
                if invalid_ratio > self.max_invalid_ratio:
                    continue

                patch_2d = stacked_2d[:, row_slice, col_slice]
                patch_target = target[:, row_slice, col_slice]
                patch_sar_mask = sar_valid_mask[row_slice, col_slice]
                patch_scl = scl[row_slice, col_slice]
                patch_has_sar = float(patch_sar_mask.mean() >= 0.50)

                validity_flags = []
                for ch in self.CANONICAL_2D_CHANNELS:
                    if ch in ("SAR_VV", "SAR_VH", "SAR_RATIO", "SAR_RVI", "MASK_SAR_VALID"):
                        validity_flags.append(patch_has_sar)
                    else:
                        validity_flags.append(1.0)
                patch_channel_validity = np.array(validity_flags, dtype=np.float32)

                patch_meta = {
                    "scene_id": str(scene_id),
                    "patch_idx": int(patch_idx),
                    "row_offset": int(r),
                    "col_offset": int(c),
                    "invalid_ratio": float(invalid_ratio),
                    "sar_coverage_ratio": float(patch_sar_mask.mean()),
                    "has_sar": int(patch_has_sar > 0.5),
                    "target_max_local": float(patch_target.max()),
                    "lat": float(metadata["lat"]),
                    "lon": float(metadata["lon"]),
                    "target_date": str(metadata["target_date"]),
                    "t0_date": str(metadata["t0_date"]),
                    "crs": str(metadata["crs"]),
                    "is_fire": int(metadata["is_fire"]),
                }

                yield {
                    "patch_id": f"{scene_id}_p{patch_idx:03d}",
                    "X_2d": patch_2d,
                    "X_1d": context_vec,
                    "channel_validity": patch_channel_validity,
                    "loss_mask": patch_loss_mask,
                    "scl": patch_scl,
                    "Y": patch_target,
                    "metadata": patch_meta,
                    "channel_names_2d": self.CANONICAL_2D_CHANNELS,
                    "context_names_1d": sorted_context_keys,
                }

                patch_idx += 1