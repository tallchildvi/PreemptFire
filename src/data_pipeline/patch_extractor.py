from typing import Any, Dict, Generator, List
import numpy as np


class PatchExtractor:
    """extracts and filters spatial patches from a master scene sample via generator streaming."""

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
        """yields valid individual patch dictionaries one by one without full-grid memory duplication."""
        rasters_2d = sample["rasters_2d"]
        loss_mask = sample["loss_mask"]
        target = sample["target"]
        context_1d = sample["context_1d"]
        metadata = sample["metadata"]

        # ensure consistent deterministic ordering of 2d channels
        channel_names: List[str] = sorted(rasters_2d.keys())
        stacked_2d = np.stack([rasters_2d[k] for k in channel_names], axis=0).astype(np.float32)

        sorted_context_keys = sorted(context_1d.keys())
        context_vec = np.array([context_1d[k] for k in sorted_context_keys], dtype=np.float32)

        _, height, width = stacked_2d.shape
        patch_idx = 0

        for r in range(0, height - self.patch_size + 1, self.stride):
            for c in range(0, width - self.patch_size + 1, self.stride):
                row_slice = slice(r, r + self.patch_size)
                col_slice = slice(c, c + self.patch_size)

                patch_loss_mask = loss_mask[row_slice, col_slice]

                # filter out patches dominated by clouds, shadows, snow, or nodata
                invalid_ratio = float((patch_loss_mask > 0.5).mean())
                if invalid_ratio > self.max_invalid_ratio:
                    continue

                patch_2d = stacked_2d[:, row_slice, col_slice]
                patch_target = target[:, row_slice, col_slice]

                # assemble comprehensive metadata for downstream spatial and temporal splitting
                patch_meta = {
                    "scene_id": str(scene_id),
                    "patch_idx": int(patch_idx),
                    "row_offset": int(r),
                    "col_offset": int(c),
                    "invalid_ratio": float(invalid_ratio),
                    "target_max_local": float(patch_target.max()),
                    "lat": float(metadata["lat"]),
                    "lon": float(metadata["lon"]),
                    "target_date": str(metadata["target_date"]),
                    "t0_date": str(metadata["t0_date"]),
                    "tprev_date": str(metadata["tprev_date"]),
                    "crs": str(metadata["crs"]),
                    "is_fire": int(metadata["is_fire"]),
                }

                yield {
                    "patch_id": f"{scene_id}_p{patch_idx:03d}",
                    "X_2d": patch_2d,
                    "X_1d": context_vec,
                    "loss_mask": patch_loss_mask,
                    "Y": patch_target,
                    "metadata": patch_meta,
                    "channel_names_2d": channel_names,
                    "context_names_1d": sorted_context_keys,
                }

                patch_idx += 1