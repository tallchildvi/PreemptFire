"""Local preparation & verification script before pushing to Hugging Face Spaces.

Run this script on your computer where you have:
- `preemptfire_full_output/` (with `models/`, `oof_predictions.csv`, `count_metrics.csv`, `summary.json`)
- `firms_points_2017_2025.csv.gz`
- (Optional) your local folder of `.npz` training scenes (`scenes/`) to extract `population_cache.csv`
  so you NEVER need Google Earth Engine credentials on Hugging Face Spaces!

Usage:
  python prepare_hf_bundle.py \
      --output-dir . \
      --models-src ../preemptfire_full_output \
      --firms-gz ../firms_points_2017_2025.csv.gz \
      --scenes-dir ../scenes
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_MODEL_FILES = [
    # 1D + History + Climatology CatBoost models
    "tab_clim_full_classifier.cbm",
    "tab_clim_full_regressor_5km.cbm",
    "tab_clim_full_regressor_10km.cbm",
    "tab_clim_full_regressor_25km.cbm",
    "tab_clim_full_regressor_50km.cbm",
    "tab_hist_classifier.cbm",
    "tab_hist_regressor_5km.cbm",
    "tab_hist_regressor_10km.cbm",
    "tab_hist_regressor_25km.cbm",
    "tab_hist_regressor_50km.cbm",
    "tab_1d_classifier.cbm",
    "tab_1d_regressor_5km.cbm",
    "tab_1d_regressor_10km.cbm",
    "tab_1d_regressor_25km.cbm",
    "tab_1d_regressor_50km.cbm",
    # 2D Satellite Image CatBoost + SimCLR models
    "tab_ssl_full_classifier.cbm",
    "tab_ssl_full_regressor_5km.cbm",
    "tab_ssl_full_regressor_10km.cbm",
    "tab_ssl_full_regressor_25km.cbm",
    "tab_ssl_full_regressor_50km.cbm",
    "simclr_encoder_weights.pt",
    "simclr_pca_32.pkl",
    "tab_full_classifier.cbm",
    "tab_full_regressor_5km.cbm",
    "tab_full_regressor_10km.cbm",
    "tab_full_regressor_25km.cbm",
    "tab_full_regressor_50km.cbm",
    # PyTorch GRU + kNN models
    "nn_ts_final_model.pt",
    "coords_classifier_knn.pkl",
    "coords_regressor_knn.pkl",
    "models_manifest.json",
]


def extract_population_cache_from_npz(metadata_csv: Path, scenes_dir: Path, out_cache: Path) -> bool:
    """Extracts exact `population_10km_log1p` and `population_50km_log1p` from local `.npz` files
    (`X_1d[14]` and `X_1d[15]`) so GEE is not needed on Hugging Face Spaces.
    """
    if not metadata_csv.exists() or not scenes_dir.exists():
        return False
    df = pd.read_csv(metadata_csv)
    la_col = next(c for c in ("lat", "latitude") if c in df.columns)
    lo_col = next(c for c in ("lon", "longitude") if c in df.columns)

    rows = []
    seen = set()
    for _, r in df.iterrows():
        lat = round(float(r[la_col]), 5)
        lon = round(float(r[lo_col]), 5)
        if (lat, lon) in seen:
            continue
        pid = str(r.get("patch_id", ""))
        npz_path = scenes_dir / f"{pid}.npz"
        if not npz_path.exists():
            continue
        try:
            z = np.load(npz_path)
            x1d = z["X_1d"].astype(np.float32)
            rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "population_10km_log1p": float(x1d[14]),
                    "population_50km_log1p": float(x1d[15]),
                }
            )
            seen.add((lat, lon))
        except Exception:
            continue

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_cache, index=False)
        print(f"[OK] Extracted population_cache.csv for {len(out_df)} unique points from {scenes_dir}")
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Prepare hf_space folder for Hugging Face Spaces deployment")
    ap.add_argument("--output-dir", default=".", help="Path to hf_space directory")
    ap.add_argument("--models-src", default=None, help="Path to preemptfire_full_output directory")
    ap.add_argument("--firms-gz", default=None, help="Path to firms_points_2017_2025.csv.gz")
    ap.add_argument("--scenes-dir", default=None, help="Optional path to local .npz scenes folder to extract population_cache.csv")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    models_dir = out_dir / "models"
    data_dir = out_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy model files and OOF/metrics from --models-src if provided
    if args.models_src:
        src = Path(args.models_src)
        src_models = src / "models" if (src / "models").exists() else src
        for f in src_models.glob("*"):
            if f.is_file():
                shutil.copy2(f, models_dir / f.name)
        for top_f in ("oof_predictions.csv", "count_metrics.csv", "summary.json"):
            cand = src / top_f
            if cand.exists():
                shutil.copy2(cand, out_dir / top_f)
                shutil.copy2(cand, data_dir / top_f)

    # 2. Copy firms_points_2017_2025.csv.gz if provided
    if args.firms_gz and Path(args.firms_gz).exists():
        shutil.copy2(args.firms_gz, out_dir / "firms_points_2017_2025.csv.gz")
        shutil.copy2(args.firms_gz, data_dir / "firms_points_2017_2025.csv.gz")
        print(f"[OK] Copied {args.firms_gz}")

    # 3. Extract population_cache.csv from local .npz scenes if available
    if args.scenes_dir:
        extract_population_cache_from_npz(
            out_dir / "metadata.csv",
            Path(args.scenes_dir),
            out_dir / "population_cache.csv",
        )

    # 4. Print verification report
    print("\n================ ПЕРЕВІРКА ФАЙЛІВ ДЛЯ HUGGING FACE SPACES ================")
    missing_models = [f for f in REQUIRED_MODEL_FILES if not (models_dir / f).exists()]
    print(f"Моделі в {models_dir}: {len(REQUIRED_MODEL_FILES) - len(missing_models)}/{len(REQUIRED_MODEL_FILES)} знайдено.")
    if missing_models:
        print("  Відсутні файли моделей (скопіюйте їх у hf_space/models/):")
        for m in missing_models:
            print(f"   - {m}")

    for df_name in ("metadata.csv", "firms_covered_days.csv", "firms_points_2017_2025.csv.gz", "oof_predictions.csv", "population_cache.csv"):
        exists = (out_dir / df_name).exists() or (data_dir / df_name).exists()
        status = "✅ Є" if exists else "⚠️ ВІДСУТНІЙ (скопіюйте в hf_space/)"
        print(f"  {df_name:32s}: {status}")


if __name__ == "__main__":
    main()
