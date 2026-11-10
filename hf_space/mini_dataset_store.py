"""Mini-dataset manager for PreemptFire.

Automatically saves:
1. 2D + 1D collected scenes in the exact repository `.npz` format (`SCENE_<lat>_<lon>_<YYYYMMDD>_p000.npz`)
   under `mini_dataset/scenes/`, plus updates `mini_dataset/metadata.csv`.
2. Daily tabular feature matrices (`TAB_1D`, `HIST_TAB`, `CLIM_FEATS`, optional `FT` & `EMB32`)
   under `mini_dataset/features/features_<YYYY-MM-DD>.csv`.
3. Daily multi-model predictions (`P(fire_50km)`, risk levels, expected counts, 90% NB upper bounds)
   under `mini_dataset/predictions/predictions_<YYYY-MM-DD>.csv`.
4. Rendered map PNGs under `mini_dataset/maps/`.
5. One-click ZIP export (`export_mini_dataset_zip`) and optional Hugging Face Dataset repo sync.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_and_features import CONFIG


class MiniDatasetStore:
    """Manages persistent storage of collected scenes, features, and predictions."""

    def __init__(self, root_dir: Optional[str | Path] = None):
        self.root = Path(root_dir or CONFIG.get("MINI_DATASET_DIR", "mini_dataset"))
        self.scenes_dir = self.root / "scenes"
        self.features_dir = self.root / "features"
        self.predictions_dir = self.root / "predictions"
        self.maps_dir = self.root / "maps"
        self.metadata_path = self.root / "metadata.csv"

        for d in (self.root, self.scenes_dir, self.features_dir, self.predictions_dir, self.maps_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_scene_id(lat: float, lon: float, target_day: date | str) -> Tuple[str, str]:
        d_str = str(target_day)[:10].replace("-", "")
        scene_id = f"SCENE_{float(lat):.4f}_{float(lon):.4f}_{d_str}"
        patch_id = f"{scene_id}_p000"
        return scene_id, patch_id

    def get_scene_npz_path(self, lat: float, lon: float, target_day: date | str) -> Path:
        _, patch_id = self.make_scene_id(lat, lon, target_day)
        return self.scenes_dir / f"{patch_id}.npz"

    def has_scene_npz(self, lat: float, lon: float, target_day: date | str, extra_dirs: Optional[List[Path]] = None) -> Optional[Path]:
        """Checks if a 2D `.npz` scene already exists in `mini_dataset/scenes` or any extra scene directories."""
        p = self.get_scene_npz_path(lat, lon, target_day)
        if p.exists():
            return p
        _, patch_id = self.make_scene_id(lat, lon, target_day)
        if extra_dirs:
            for d in extra_dirs:
                cand = Path(d) / f"{patch_id}.npz"
                if cand.exists():
                    return cand
        # Also look for matching lat/lon (within 1e-3 deg) on the same date in scenes_dir
        d_str = str(target_day)[:10].replace("-", "")
        for f in self.scenes_dir.glob(f"SCENE_*_{d_str}_p000.npz"):
            parts = f.stem.split("_")
            if len(parts) >= 4:
                try:
                    f_lat, f_lon = float(parts[1]), float(parts[2])
                    if abs(f_lat - float(lat)) < 1e-3 and abs(f_lon - float(lon)) < 1e-3:
                        return f
                except ValueError:
                    continue
        return None

    def load_scene_npz(self, npz_path: str | Path) -> Dict[str, Any]:
        z = np.load(npz_path, allow_pickle=True)
        out = {
            "X_2d": z["X_2d"].astype(np.float32),
            "scl": z["scl"].astype(np.uint8),
            "X_1d": z["X_1d"].astype(np.float32),
            "channel_validity": z["channel_validity"].astype(np.float32)
            if "channel_validity" in z.files
            else np.ones(22, dtype=np.float32),
            "npz_path": str(npz_path),
        }
        return out

    def save_scene_npz(self, scene_data: Dict[str, Any]) -> Path:
        """Saves a collected 2D+1D scene into `mini_dataset/scenes/<patch_id>.npz` and updates `mini_dataset/metadata.csv`."""
        lat = float(scene_data["latitude"])
        lon = float(scene_data["longitude"])
        date_str = str(scene_data["date"])[:10]
        scene_id, patch_id = self.make_scene_id(lat, lon, date_str)
        npz_path = self.scenes_dir / f"{patch_id}.npz"

        X_2d = np.asarray(scene_data["X_2d"], dtype=np.float16)
        scl = np.asarray(scene_data["scl"], dtype=np.uint8)
        X_1d = np.asarray(scene_data["X_1d"], dtype=np.float32)
        channel_validity = np.asarray(
            scene_data.get("channel_validity", np.ones(22, dtype=np.float32)),
            dtype=np.float32,
        )

        np.savez_compressed(
            npz_path,
            X_2d=X_2d,
            scl=scl,
            X_1d=X_1d,
            channel_validity=channel_validity,
        )

        row = {
            "patch_id": patch_id,
            "scene_id": scene_id,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "date": date_str,
            "is_fire_scene": scene_data.get("is_fire_scene", False),
            "fire_pixel_ratio": float(scene_data.get("fire_pixel_ratio", 0.0)),
            "s2_Source_Date": scene_data.get("s2_Source_Date", "None"),
            "valid_s2_frac": float(scene_data.get("valid_s2_frac", 0.0)),
            "collected_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        # Merge any predictions attached to scene_data
        if "predictions" in scene_data and isinstance(scene_data["predictions"], dict):
            for k, v in scene_data["predictions"].items():
                if isinstance(v, (int, float, str, bool, np.number)):
                    row[k] = float(v) if isinstance(v, np.number) else v

        if self.metadata_path.exists():
            try:
                df = pd.read_csv(self.metadata_path)
                df = df[df["patch_id"] != patch_id]
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            except Exception:
                df = pd.DataFrame([row])
        else:
            df = pd.DataFrame([row])
        df.to_csv(self.metadata_path, index=False)
        return npz_path

    def save_daily_features(self, target_day: date | str, features_df: pd.DataFrame) -> Path:
        d_str = str(target_day)[:10]
        out_path = self.features_dir / f"features_{d_str}.csv"
        features_df.to_csv(out_path, index=False)
        return out_path

    def load_daily_features(self, target_day: date | str) -> Optional[pd.DataFrame]:
        d_str = str(target_day)[:10]
        p = self.features_dir / f"features_{d_str}.csv"
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None

    def save_daily_predictions(self, target_day: date | str, preds_df: pd.DataFrame, meta_summary: Optional[Dict[str, Any]] = None) -> Path:
        d_str = str(target_day)[:10]
        out_path = self.predictions_dir / f"predictions_{d_str}.csv"
        preds_df.to_csv(out_path, index=False)
        if meta_summary is not None:
            json_path = self.predictions_dir / f"summary_{d_str}.json"
            json_path.write_text(json.dumps(meta_summary, indent=2, ensure_ascii=False))
        return out_path

    def summary_stats(self) -> Dict[str, Any]:
        npz_files = sorted(self.scenes_dir.glob("*.npz"))
        feat_files = sorted(self.features_dir.glob("*.csv"))
        pred_files = sorted(self.predictions_dir.glob("predictions_*.csv"))
        map_files = sorted(self.maps_dir.glob("*.png"))
        total_bytes = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return {
            "root_dir": str(self.root.resolve()),
            "num_npz_scenes": len(npz_files),
            "num_feature_days": len(feat_files),
            "num_prediction_days": len(pred_files),
            "num_saved_maps": len(map_files),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "npz_files": [f.name for f in npz_files[-50:]],
            "feature_files": [f.name for f in feat_files[-20:]],
            "prediction_files": [f.name for f in pred_files[-20:]],
        }

    def export_zip(self, zip_path: Optional[str | Path] = None) -> Path:
        """Packages the entire `mini_dataset/` folder into a downloadable ZIP archive."""
        if zip_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            zip_path = self.root.parent / f"preemptfire_mini_dataset_{ts}.zip"
        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(self.root.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.root.parent)
                    zf.write(file_path, arcname)
        return zip_path
