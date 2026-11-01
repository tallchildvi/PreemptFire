"""
src/modeling/eda.py

Performs rapid audit and exploratory data analysis of the consolidated HDF5 dataset.
Inspects the 27 canonical X_2d channels, 38 X_1d context features, Y targets,
and JSON metadata records.

Usage:
    python -m src.modeling.eda data/processed/dataset.h5
"""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def run_eda(h5_path: str) -> None:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    print("=" * 65)
    print(f"[INFO] INSPECTING DATASET: {path.resolve()}")
    print("=" * 65)

    with h5py.File(path, "r") as f:
        print(f"Root HDF5 Datasets: {list(f.keys())}")

        total_patches = f["X_2d"].shape[0]
        print(f"\nTotal Patches:           {total_patches}")
        print(f"X_2d Tensor Shape:       {f['X_2d'].shape} | dtype: {f['X_2d'].dtype}")
        print(f"X_1d Vector Shape:       {f['X_1d'].shape} | dtype: {f['X_1d'].dtype}")
        print(f"Target (Y) Shape:        {f['Y'].shape}    | dtype: {f['Y'].dtype}")
        print(f"Channel Validity Shape:  {f['channel_validity'].shape} | dtype: {f['channel_validity'].dtype}")

        # Parse JSON metadata strings to audit class balance and unique scenes
        print("\n[INFO] Parsing metadata records...")
        raw_meta = f["metadata"][:]
        is_fire_arr = np.zeros(total_patches, dtype=np.int32)
        has_sar_arr = np.zeros(total_patches, dtype=np.int32)
        scenes = set()

        for i, m in enumerate(raw_meta):
            m_str = m.decode("utf-8") if isinstance(m, bytes) else str(m)
            d = json.loads(m_str)
            is_fire_arr[i] = int(d.get("is_fire", 0))
            has_sar_arr[i] = int(d.get("has_sar", 0))
            scenes.add(d.get("scene_id", "unknown"))

        pos = int(np.sum(is_fire_arr == 1))
        neg = int(np.sum(is_fire_arr == 0))
        sar_count = int(np.sum(has_sar_arr == 1))

        print("\n--- CLASS DISTRIBUTION & SENSOR COVERAGE (from metadata) ---")
        print(f"Fire Patches (is_fire=1):       {pos:6d} ({pos / total_patches * 100:.2f}%)")
        print(f"Background Patches (is_fire=0): {neg:6d} ({neg / total_patches * 100:.2f}%)")
        print(f"Patches with Valid SAR:         {sar_count:6d} ({sar_count / total_patches * 100:.2f}%)")
        print(f"Unique Parent Scenes:           {len(scenes):6d} (Avg {total_patches / max(len(scenes), 1):.1f} patches/scene)")

        # Numerical integrity check on first 100 patches
        sample_size = min(100, total_patches)
        x2d_sample = f["X_2d"][:sample_size].astype(np.float32)
        x1d_sample = f["X_1d"][:sample_size].astype(np.float32)

        # Combine masks (22: Water, 24: Clouds, 25: Cloud Shadows) on sample
        combined_invalid = (
            (x2d_sample[:, 22] > 0.5)
            | (x2d_sample[:, 24] > 0.5)
            | (x2d_sample[:, 25] > 0.5)
        )
        valid_pixel_ratio = float((~combined_invalid).mean()) * 100.0

        print(f"\n--- NUMERICAL & MASK AUDIT (First {sample_size} patches) ---")
        print(f"X_2d NaN count:          {int(np.isnan(x2d_sample).sum())}")
        print(f"X_2d Inf count:          {int(np.isinf(x2d_sample).sum())}")
        print(f"X_1d NaN count:          {int(np.isnan(x1d_sample).sum())}")
        print(f"X_1d Inf count:          {int(np.isinf(x1d_sample).sum())}")
        print(f"Valid Land Pixel Ratio:  {valid_pixel_ratio:.2f}% (excluding water, clouds, shadows)")

        if "channel_names_2d" in f.attrs:
            c2d = json.loads(f.attrs["channel_names_2d"])
            print(f"\n--- CANONICAL 2D CHANNELS ({len(c2d)} channels) ---")
            print(", ".join(f"{idx}:{name}" for idx, name in enumerate(c2d)))

        if "context_names_1d" in f.attrs:
            c1d = json.loads(f.attrs["context_names_1d"])
            print(f"\n--- CONTEXT 1D FEATURES ({len(c1d)} features) ---")
            print(", ".join(c1d))

    print("\n[SUCCESS] Dataset inspection completed without errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit wildfire HDF5 dataset.")
    parser.add_argument("h5_path", nargs="?", default="data/processed/dataset.h5")
    args = parser.parse_args()
    run_eda(args.h5_path)