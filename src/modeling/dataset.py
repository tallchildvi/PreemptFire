import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import DataLoader, Dataset


class WildfireDataset(Dataset):
    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: Optional[List[int]] = None,
        compute_indices: bool = True,
        transform=None,
    ):
        self.h5_path = str(h5_path)
        self.compute_indices = compute_indices
        self.transform = transform
        self._h5_file = None

        with h5py.File(self.h5_path, "r") as f:
            total_len = f["X_2d"].shape[0]
            if "channel_names_2d" in f.attrs:
                ch_names = json.loads(f.attrs["channel_names_2d"])
                self.ch_map = {name: i for i, name in enumerate(ch_names)}
            else:
                self.ch_map = {
                    "B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4,
                    "B07": 5, "B08": 6, "B8A": 7, "B11": 8, "B12": 9,
                    "SAR_VV": 10, "SAR_VH": 11, "Elevation": 12, "Slope": 13,
                    "Northness": 14, "Eastness": 15, "Travel_Time_Roads": 16,
                    "Travel_Time_Trails": 17, "Dist_to_Railways": 18,
                    "Dist_to_Camps": 19, "Dist_to_Powerlines": 20, "Soil_Moisture": 21,
                    "MASK_WATER": 22, "MASK_SNOW": 23, "MASK_CLOUDS": 24,
                    "MASK_CLOUD_SHADOWS": 25, "MASK_SAR_VALID": 26
                }

        self.indices = list(range(total_len)) if indices is None else indices

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return len(self.indices)

    def _calc_indices_on_the_fly(self, x2d: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        b04 = x2d[self.ch_map["B04"]]
        b05 = x2d[self.ch_map["B05"]]
        b08 = x2d[self.ch_map["B08"]]
        b11 = x2d[self.ch_map["B11"]]
        b12 = x2d[self.ch_map["B12"]]

        ndvi = (b08 - b04) / (b08 + b04 + eps)
        ndmi = (b08 - b11) / (b08 + b11 + eps)
        nbr2 = (b11 - b12) / (b11 + b12 + eps)
        ndre = (b08 - b05) / (b08 + b05 + eps)

        extra = np.stack([ndvi, ndmi, nbr2, ndre], axis=0).astype(np.float32)
        extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([x2d, extra], axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5 = self._get_h5()
        real_idx = self.indices[idx]

        x_2d = h5["X_2d"][real_idx].astype(np.float32)
        x_1d = h5["X_1d"][real_idx].astype(np.float32)
        y = h5["Y"][real_idx].astype(np.float32)
        loss_mask = h5["loss_mask"][real_idx].astype(np.float32)
        is_fire = float(h5["is_fire"][real_idx])

        if self.compute_indices:
            x_2d = self._calc_indices_on_the_fly(x_2d)

        x_2d = np.nan_to_num(x_2d, nan=0.0)
        x_1d = np.nan_to_num(x_1d, nan=0.0)

        item = {
            "x_2d": torch.from_numpy(x_2d),
            "x_1d": torch.from_numpy(x_1d),
            "target": torch.from_numpy(y),
            "loss_mask": torch.from_numpy(loss_mask),
            "is_fire": torch.tensor(is_fire, dtype=torch.float32),
            "index": real_idx,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


def get_spatial_split_indices(h5_path: str, n_splits: int = 5, test_fold: int = 0) -> Tuple[List[int], List[int]]:
    with h5py.File(h5_path, "r") as f:
        scene_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["scene_id"][:]]
        is_fire = f["is_fire"][:]

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X=np.arange(len(scene_ids)), y=is_fire, groups=scene_ids))

    train_idx, val_idx = splits[test_fold]
    return train_idx.tolist(), val_idx.tolist()