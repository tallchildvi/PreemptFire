from typing import Any, Dict, Generator, List
import numpy as np


class PatchExtractor:
    """extracts and filters spatial patches from a master scene sample via generator streaming."""

    # fixed contract of all 2d channels
    CANONICAL_2D_CHANNELS: List[str] = [
        # optical bands (t0)
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
        # spectral indices
        "EVI_T0", "MSI_T0", "NBR_T0", "NBR2_T0", "NDMI_T0", "NDRE_T0", "NDVI_T0", "NMDI_T0",
        "dNBR", "dNDMI", "dNDVI", "dNMDI", "dMSI",
        # sentinel-1 sar and polarimetric indices
        "SAR_VV", "SAR_VH", "SAR_RATIO", "SAR_RVI",
        # digital elevation model features
        "Elevation", "Slope", "Northness", "Eastness",
        # spatial accessibility and proximity features
        "Travel_Time_Roads", "Travel_Time_Trails",
        "Dist_to_Railways", "Dist_to_Camps", "Dist_to_Powerlines",
        # socio-economic and soil moisture features
        "Nightlight_Potential", "Population_Potential", "Soil_Moisture",
        # optical quality masks
        "MASK_WATER", "MASK_SNOW", "MASK_CLOUDS", "MASK_CLOUD_SHADOWS",
        # modality validity indicator
        "MASK_SAR_VALID",
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
        """yields valid individual patch dictionaries one by one without full-grid memory duplication."""
        rasters_2d = sample["rasters_2d"]
        loss_mask = sample["loss_mask"]
        target = sample["target"]
        context_1d = sample["context_1d"]
        metadata = sample["metadata"]

        h_full, w_full = loss_mask.shape

        # determine sar availability and build validity channel
        has_sar = ("SAR_VV" in rasters_2d and "SAR_VH" in rasters_2d)
        sar_valid_mask = np.full((h_full, w_full), 1.0 if has_sar else 0.0, dtype=np.float32)

        # assemble 2d channels according to canonical schema with zero-imputation
        canonical_rasters = []
        for ch in self.CANONICAL_2D_CHANNELS:
            if ch == "MASK_SAR_VALID":
                canonical_rasters.append(sar_valid_mask)
            elif ch in rasters_2d and rasters_2d[ch] is not None:
                canonical_rasters.append(np.nan_to_num(rasters_2d[ch], nan=0.0).astype(np.float32))
            else:
                # impute missing modality with zero baseline to maintain fixed tensor dimensions
                canonical_rasters.append(np.zeros((h_full, w_full), dtype=np.float32))

        stacked_2d = np.stack(canonical_rasters, axis=0).astype(np.float32)

        # sort 1d keys for deterministic feature vector creation
        sorted_context_keys = sorted(context_1d.keys())
        context_vec = np.array([float(context_1d[k]) for k in sorted_context_keys], dtype=np.float32)

        patch_idx = 0

        for r in range(0, h_full - self.patch_size + 1, self.stride):
            for c in range(0, w_full - self.patch_size + 1, self.stride):
                row_slice = slice(r, r + self.patch_size)
                col_slice = slice(c, c + self.patch_size)

                patch_loss_mask = loss_mask[row_slice, col_slice]

                # filter out patches dominated by clouds, shadows, snow, or nodata
                invalid_ratio = float((patch_loss_mask > 0.5).mean())
                if invalid_ratio > self.max_invalid_ratio:
                    continue

                patch_2d = stacked_2d[:, row_slice, col_slice]
                patch_target = target[:, row_slice, col_slice]

                # assemble comprehensive metadata
                patch_meta = {
                    "scene_id": str(scene_id),
                    "patch_idx": int(patch_idx),
                    "row_offset": int(r),
                    "col_offset": int(c),
                    "invalid_ratio": float(invalid_ratio),
                    "has_sar": int(has_sar),
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
                    "channel_names_2d": self.CANONICAL_2D_CHANNELS,
                    "context_names_1d": sorted_context_keys,
                }

                patch_idx += 1