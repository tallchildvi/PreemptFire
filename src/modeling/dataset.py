import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import Dataset


class IndexCalculator:
    """Computes spectral vegetation, fuel moisture, and SAR polarimetric indices."""

    @staticmethod
    def calc_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Vegetation Index: (B08 - B04) / (B08 + B04)."""
        return ((nir - red) / (nir + red + eps)).astype(np.float32)

    @staticmethod
    def calc_ndmi(nir: np.ndarray, swir1: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Moisture Index: (B08 - B11) / (B08 + B11)."""
        return ((nir - swir1) / (nir + swir1 + eps)).astype(np.float32)

    @staticmethod
    def calc_nbr(nir: np.ndarray, swir2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Burn Ratio: (B08 - B12) / (B08 + B12)."""
        return ((nir - swir2) / (nir + swir2 + eps)).astype(np.float32)

    @staticmethod
    def calc_nbr2(swir1: np.ndarray, swir2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Burn Ratio 2: (B11 - B12) / (B11 + B12)."""
        return ((swir1 - swir2) / (swir1 + swir2 + eps)).astype(np.float32)

    @staticmethod
    def calc_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Enhanced Vegetation Index: 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)."""
        numerator = 2.5 * (nir - red)
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0
        safe_denom = np.where(
            np.abs(denominator) < 1e-3,
            np.where(denominator >= 0, 1e-3, -1e-3),
            denominator,
        )
        return np.clip(numerator / safe_denom, -2.0, 2.0).astype(np.float32)

    @staticmethod
    def calc_ndre(nir: np.ndarray, red_edge: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Red Edge Index: (B08 - B05) / (B08 + B05)."""
        return ((nir - red_edge) / (nir + red_edge + eps)).astype(np.float32)

    @staticmethod
    def calc_msi(swir1: np.ndarray, nir: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Moisture Stress Index: B11 / B08."""
        return np.clip((swir1 + eps) / (nir + eps), 0.0, 10.0).astype(np.float32)

    @staticmethod
    def calc_nmdi(
        nir: np.ndarray,
        swir1: np.ndarray,
        swir2: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Normalized Multi-band Drought Index: (B08 - (B11 - B12)) / (B08 + (B11 - B12))."""
        diff_swir = swir1 - swir2
        numerator = nir - diff_swir
        denominator = nir + diff_swir
        safe_denom = np.where(
            np.abs(denominator) < 1e-3,
            np.where(denominator >= 0, 1e-3, -1e-3),
            denominator,
        )
        return np.clip(numerator / safe_denom, -2.0, 2.0).astype(np.float32)

    @staticmethod
    def calc_sar_ratio(vh: np.ndarray, vv: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Cross-polarization ratio: VH / VV."""
        return ((vh + eps) / (vv + eps)).astype(np.float32)

    @staticmethod
    def calc_sar_rvi(vv: np.ndarray, vh: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Radar Vegetation Index: 4 * VH / (VV + VH)."""
        return ((4.0 * vh) / (vv + vh + eps)).astype(np.float32)


class WildfireDataset(Dataset):
    """
    PyTorch Dataset with lazy SWMR HDF5 initialization.
    Synthesizes 10 spectral/SAR indices and builds loss_mask directly from X_2d mask channels.
    """

    CANONICAL_CHANNELS = {
        "B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4,
        "B07": 5, "B08": 6, "B8A": 7, "B11": 8, "B12": 9,
        "SAR_VV": 10, "SAR_VH": 11, "Elevation": 12, "Slope": 13,
        "Northness": 14, "Eastness": 15, "Travel_Time_Roads": 16,
        "Travel_Time_Trails": 17, "Dist_to_Railways": 18,
        "Dist_to_Camps": 19, "Dist_to_Powerlines": 20, "Soil_Moisture": 21,
        "MASK_WATER": 22, "MASK_SNOW": 23, "MASK_CLOUDS": 24,
        "MASK_CLOUD_SHADOWS": 25, "MASK_SAR_VALID": 26,
    }

    DYNAMIC_INDEX_NAMES = [
        "NDVI", "NDMI", "NBR", "NBR2", "EVI",
        "NDRE", "MSI", "NMDI", "SAR_RATIO", "SAR_RVI",
    ]

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: Optional[List[int]] = None,
        compute_indices: bool = True,
        mask_snow_in_loss: bool = False,
        transform: Optional[Callable] = None,
        use_coords: bool = False,
    ):
        # Initialize _h5_file first to prevent AttributeError in __del__ if __init__ fails
        self._h5_file = None
        self.h5_path = str(Path(h5_path))
        self.compute_indices = compute_indices
        self.mask_snow_in_loss = mask_snow_in_loss
        self.transform = transform
        # use_coords=True appends [lat_norm, lon_norm] to x_1d (38 -> 40 features),
        # so the 1D branch of the model must be built with in_dim=40.
        self.use_coords = use_coords

        with h5py.File(self.h5_path, "r") as f:
            total_patches = f["X_2d"].shape[0]
            if "channel_names_2d" in f.attrs:
                ch_names = json.loads(f.attrs["channel_names_2d"])
                self.ch_map = {name: i for i, name in enumerate(ch_names)}
            else:
                self.ch_map = dict(self.CANONICAL_CHANNELS)

        self.indices = list(range(total_patches)) if indices is None else indices

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return len(self.indices)

    def _build_loss_mask(self, x2d: np.ndarray) -> np.ndarray:
        """
        Combines binary mask channels from X_2d into a unified valid-pixel loss mask.
        Returns 1.0 for valid land pixels, 0.0 for masked out pixels (water, clouds, shadows).
        """
        mask_water = x2d[self.ch_map["MASK_WATER"]] > 0.5
        mask_clouds = x2d[self.ch_map["MASK_CLOUDS"]] > 0.5
        mask_shadows = x2d[self.ch_map["MASK_CLOUD_SHADOWS"]] > 0.5

        invalid = mask_water | mask_clouds | mask_shadows
        if self.mask_snow_in_loss:
            mask_snow = x2d[self.ch_map["MASK_SNOW"]] > 0.5
            invalid = invalid | mask_snow

        valid_mask = (~invalid).astype(np.float32)
        return valid_mask

    def _synthesize_indices(self, x2d: np.ndarray) -> np.ndarray:
        """Calculates 10 dynamic indices and appends them to the 27 base channels (total: 37)."""
        blue = x2d[self.ch_map["B02"]]
        red = x2d[self.ch_map["B04"]]
        red_edge = x2d[self.ch_map["B05"]]
        nir = x2d[self.ch_map["B08"]]
        swir1 = x2d[self.ch_map["B11"]]
        swir2 = x2d[self.ch_map["B12"]]
        vv = x2d[self.ch_map["SAR_VV"]]
        vh = x2d[self.ch_map["SAR_VH"]]

        ndvi = IndexCalculator.calc_ndvi(nir, red)
        ndmi = IndexCalculator.calc_ndmi(nir, swir1)
        nbr = IndexCalculator.calc_nbr(nir, swir2)
        nbr2 = IndexCalculator.calc_nbr2(swir1, swir2)
        evi = IndexCalculator.calc_evi(nir, red, blue)
        ndre = IndexCalculator.calc_ndre(nir, red_edge)
        msi = IndexCalculator.calc_msi(swir1, nir)
        nmdi = IndexCalculator.calc_nmdi(nir, swir1, swir2)
        sar_ratio = IndexCalculator.calc_sar_ratio(vh, vv)
        sar_rvi = IndexCalculator.calc_sar_rvi(vv, vh)

        extra = np.stack([
            ndvi, ndmi, nbr, nbr2, evi,
            ndre, msi, nmdi, sar_ratio, sar_rvi,
        ], axis=0).astype(np.float32)

        extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([x2d, extra], axis=0)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        h5 = self._get_h5()
        real_idx = self.indices[idx]

        # Convert float16 storage arrays to float32 for numerical stability
        x_2d = h5["X_2d"][real_idx].astype(np.float32)
        x_1d = h5["X_1d"][real_idx].astype(np.float32)
        y = h5["Y"][real_idx].astype(np.float32)
        channel_validity = h5["channel_validity"][real_idx].astype(np.float32)

        # Parse JSON metadata string to extract is_fire flag
        raw_meta = h5["metadata"][real_idx]
        meta_str = raw_meta.decode("utf-8") if isinstance(raw_meta, bytes) else str(raw_meta)
        meta_dict = json.loads(meta_str)
        is_fire = float(meta_dict.get("is_fire", 0))

        # Patch coordinates from metadata (WGS84 degrees). Missing -> NaN -> 0 after nan_to_num.
        lat = float(meta_dict.get("lat", float("nan")))
        lon = float(meta_dict.get("lon", float("nan")))
        coords_raw = np.array([lat, lon], dtype=np.float32)
        # Rough standardisation for the Iberian Peninsula (lat ~36-44, lon ~-10..4)
        coords_norm = np.array([(lat - 40.0) / 4.0, (lon + 4.0) / 5.0], dtype=np.float32)

        # Build composite loss mask from MASK_WATER, MASK_CLOUDS, MASK_CLOUD_SHADOWS
        loss_mask = self._build_loss_mask(x_2d)

        # Synthesize 10 spectral and radar indices on-the-fly
        if self.compute_indices:
            x_2d = self._synthesize_indices(x_2d)

        x_2d = np.nan_to_num(x_2d, nan=0.0, posinf=0.0, neginf=0.0)
        x_1d = np.nan_to_num(x_1d, nan=0.0, posinf=0.0, neginf=0.0)
        coords_norm = np.nan_to_num(coords_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if self.use_coords:
            x_1d = np.concatenate([x_1d, coords_norm]).astype(np.float32)

        item = {
            "coords": torch.from_numpy(coords_norm),
            "coords_raw": torch.from_numpy(coords_raw),
            "x_2d": torch.from_numpy(x_2d),
            "x_1d": torch.from_numpy(x_1d),
            "target": torch.from_numpy(y),
            "loss_mask": torch.from_numpy(loss_mask),
            "channel_validity": torch.from_numpy(channel_validity),
            "is_fire": torch.tensor(is_fire, dtype=torch.float32),
            "index": real_idx,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def close(self) -> None:
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:
        self.close()


def get_spatial_split_indices(
    h5_path: Union[str, Path],
    n_splits: int = 5,
    test_fold: int = 0,
    group_by: str = "scene",
    cell_deg: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    GroupKFold splits parsed from metadata.

    group_by="scene": groups = scene_id (prevents leakage between patches of one scene).
    group_by="cell":  groups = cell_deg x cell_deg lat/lon cells. Use this whenever lat/lon
                      are model inputs: otherwise neighbouring scenes (e.g. a fire scene and
                      its hard negative ~60 km away) land in train and val, and the model can
                      memorise "this location" instead of learning risk.
    """
    with h5py.File(h5_path, "r") as f:
        md = f["metadata"]
        raw_meta_all = []
        for s in range(0, md.shape[0], 500):  # chunked read: a single [:] can stall on slow storage
            raw_meta_all.extend(md[s:s + 500])

    groups = []
    is_fire_list = []
    for raw_m in raw_meta_all:
        m_str = raw_m.decode("utf-8") if isinstance(raw_m, bytes) else str(raw_m)
        m_dict = json.loads(m_str)
        if group_by == "cell":
            groups.append(f"{np.floor(m_dict['lat'] / cell_deg):.0f}_{np.floor(m_dict['lon'] / cell_deg):.0f}")
        else:
            groups.append(str(m_dict["scene_id"]))
        is_fire_list.append(int(m_dict["is_fire"]))

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X=np.arange(len(groups)), y=is_fire_list, groups=groups))
    return splits[test_fold][0].tolist(), splits[test_fold][1].tolist()