"""
src/modeling/train_catboost.py

Tabular Baseline Pipeline for Wildfire Susceptibility Forecasting.
Extracts the 38 1D context features and aggregates the 27 canonical 2D channels
plus 10 dynamically synthesized spectral/radar indices over valid land pixels.
Trains and evaluates a CatBoost model using spatial GroupKFold validation.

Usage:
    python -m src.modeling.train_catboost --h5 data/processed/dataset.h5
"""

import argparse
import json
from pathlib import Path
from catboost import CatBoostClassifier, Pool
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.modeling.dataset import IndexCalculator, WildfireDataset


def extract_tabular_dataset(h5_path: str, chunk_size: int = 256) -> pd.DataFrame:
    """
    Streams chunks from HDF5, computes 10 physical indices, and aggregates spatial
    statistics across valid pixels (excluding clouds, shadows, and water).
    """
    print(f"[INFO] Extracting tabular features from: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        n_samples = f["X_2d"].shape[0]

        if "context_names_1d" in f.attrs:
            c1d_names = json.loads(f.attrs["context_names_1d"])
        else:
            c1d_names = [f"feat_1d_{i}" for i in range(f["X_1d"].shape[1])]

        if "channel_names_2d" in f.attrs:
            c2d_names = json.loads(f.attrs["channel_names_1d" if "channel_names_2d" not in f.attrs else "channel_names_2d"])
            c2d_map = {name: i for i, name in enumerate(c2d_names)}
        else:
            c2d_map = dict(WildfireDataset.CANONICAL_CHANNELS)

        all_chunks = []

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)

            # Load chunk and cast float16 rasters to float32
            batch_x1d = f["X_1d"][start_idx:end_idx].astype(np.float32)
            batch_x2d = f["X_2d"][start_idx:end_idx].astype(np.float32)
            batch_meta = f["metadata"][start_idx:end_idx]

            # Parse metadata JSON strings for target label and scene grouping
            batch_is_fire = []
            batch_scene = []
            for raw_m in batch_meta:
                m_str = raw_m.decode("utf-8") if isinstance(raw_m, bytes) else str(raw_m)
                m_dict = json.loads(m_str)
                batch_is_fire.append(int(m_dict.get("is_fire", 0)))
                batch_scene.append(str(m_dict.get("scene_id", "unknown")))

            chunk_df = pd.DataFrame(batch_x1d, columns=c1d_names)

            # Extract base bands from the 27 canonical channels
            blue = batch_x2d[:, c2d_map["B02"]]
            red = batch_x2d[:, c2d_map["B04"]]
            red_edge = batch_x2d[:, c2d_map["B05"]]
            nir = batch_x2d[:, c2d_map["B08"]]
            swir1 = batch_x2d[:, c2d_map["B11"]]
            swir2 = batch_x2d[:, c2d_map["B12"]]
            vv = batch_x2d[:, c2d_map["SAR_VV"]]
            vh = batch_x2d[:, c2d_map["SAR_VH"]]

            # Extract quality and coverage masks (channels 22..26)
            mask_water = batch_x2d[:, c2d_map["MASK_WATER"]]
            mask_snow = batch_x2d[:, c2d_map["MASK_SNOW"]]
            mask_clouds = batch_x2d[:, c2d_map["MASK_CLOUDS"]]
            mask_shadows = batch_x2d[:, c2d_map["MASK_CLOUD_SHADOWS"]]
            mask_sar_valid = batch_x2d[:, c2d_map["MASK_SAR_VALID"]]

            # Compute 10 dynamic indices
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

            # 1. Optical vegetation and fuel moisture summaries
            chunk_df["agg_ndvi_mean"] = np.nanmean(ndvi, axis=(1, 2))
            chunk_df["agg_ndvi_min"] = np.nanmin(ndvi, axis=(1, 2))
            chunk_df["agg_evi_mean"] = np.nanmean(evi, axis=(1, 2))
            chunk_df["agg_ndre_mean"] = np.nanmean(ndre, axis=(1, 2))

            chunk_df["agg_ndmi_mean"] = np.nanmean(ndmi, axis=(1, 2))
            chunk_df["agg_ndmi_min"] = np.nanmin(ndmi, axis=(1, 2))
            chunk_df["agg_nbr_mean"] = np.nanmean(nbr, axis=(1, 2))
            chunk_df["agg_nbr2_mean"] = np.nanmean(nbr2, axis=(1, 2))
            chunk_df["agg_msi_mean"] = np.nanmean(msi, axis=(1, 2))
            chunk_df["agg_msi_max"] = np.nanmax(msi, axis=(1, 2))
            chunk_df["agg_nmdi_mean"] = np.nanmean(nmdi, axis=(1, 2))

            # 2. Radar backscatter and polarimetric ratios
            chunk_df["agg_sar_vv_mean"] = np.nanmean(vv, axis=(1, 2))
            chunk_df["agg_sar_vh_mean"] = np.nanmean(vh, axis=(1, 2))
            chunk_df["agg_sar_ratio_mean"] = np.nanmean(sar_ratio, axis=(1, 2))
            chunk_df["agg_sar_rvi_mean"] = np.nanmean(sar_rvi, axis=(1, 2))

            # 3. Topography & Soil Moisture (channels 12..15, 21)
            chunk_df["agg_elevation_mean"] = np.nanmean(batch_x2d[:, c2d_map["Elevation"]], axis=(1, 2))
            chunk_df["agg_slope_mean"] = np.nanmean(batch_x2d[:, c2d_map["Slope"]], axis=(1, 2))
            chunk_df["agg_slope_max"] = np.nanmax(batch_x2d[:, c2d_map["Slope"]], axis=(1, 2))
            chunk_df["agg_northness_mean"] = np.nanmean(batch_x2d[:, c2d_map["Northness"]], axis=(1, 2))
            chunk_df["agg_eastness_mean"] = np.nanmean(batch_x2d[:, c2d_map["Eastness"]], axis=(1, 2))
            chunk_df["agg_soil_moisture_mean"] = np.nanmean(batch_x2d[:, c2d_map["Soil_Moisture"]], axis=(1, 2))
            chunk_df["agg_soil_moisture_min"] = np.nanmin(batch_x2d[:, c2d_map["Soil_Moisture"]], axis=(1, 2))

            # 4. Anthropogenic access and proximity (channels 16..20)
            for osm_col in [
                "Travel_Time_Roads",
                "Travel_Time_Trails",
                "Dist_to_Railways",
                "Dist_to_Camps",
                "Dist_to_Powerlines",
            ]:
                arr_osm = batch_x2d[:, c2d_map[osm_col]]
                chunk_df[f"agg_min_{osm_col}"] = np.nanmin(arr_osm, axis=(1, 2))
                chunk_df[f"agg_mean_{osm_col}"] = np.nanmean(arr_osm, axis=(1, 2))

            # 5. Mask coverage fractions (channels 22..26)
            chunk_df["frac_water"] = np.mean(mask_water > 0.5, axis=(1, 2))
            chunk_df["frac_snow"] = np.mean(mask_snow > 0.5, axis=(1, 2))
            chunk_df["frac_clouds"] = np.mean((mask_clouds > 0.5) | (mask_shadows > 0.5), axis=(1, 2))
            chunk_df["frac_sar_valid"] = np.mean(mask_sar_valid > 0.5, axis=(1, 2))

            chunk_df["is_fire"] = batch_is_fire
            chunk_df["scene_id"] = batch_scene
            all_chunks.append(chunk_df)

            if end_idx % 2048 == 0 or end_idx == n_samples:
                print(f"[PROGRESS] Processed {end_idx}/{n_samples} patches...")

    full_df = pd.concat(all_chunks, ignore_index=True)
    return full_df


# Columns that describe the sample but must never be used as features
META_COLS = ["is_fire", "scene_id", "target_date", "spatial_group"]
COORD_COLS = ["lat", "lon"]
DOY_COLS = ["doy_sin", "doy_cos"]


def add_meta_features(df: pd.DataFrame, h5_path: str, cell_deg: float = 0.5) -> pd.DataFrame:
    """
    Adds lat, lon, target_date, doy_sin/doy_cos and a spatial_group (cell_deg x cell_deg cell)
    to an already extracted feature table, reading ONLY the metadata (fast, no rasters).
    Works with an old cached parquet that has no coordinates.
    """
    rows = []
    with h5py.File(h5_path, "r") as f:
        md = f["metadata"]
        if md.shape[0] != len(df):
            raise ValueError(f"metadata rows ({md.shape[0]}) != cached table rows ({len(df)}): rebuild the cache")
        for s in range(0, md.shape[0], 500):  # chunked: a single md[:] can stall on slow storage
            for raw_m in md[s:s + 500]:
                m_str = raw_m.decode("utf-8") if isinstance(raw_m, bytes) else str(raw_m)
                m = json.loads(m_str)
                rows.append((str(m["scene_id"]), float(m["lat"]), float(m["lon"]), str(m["target_date"])))
    meta = pd.DataFrame(rows, columns=["scene_id", "lat", "lon", "target_date"])
    if not (meta["scene_id"].values == df["scene_id"].values).all():
        raise ValueError("scene_id order differs between cache and HDF5: rebuild the cache")

    out = df.copy()
    out["lat"] = meta["lat"].values
    out["lon"] = meta["lon"].values
    out["target_date"] = meta["target_date"].values
    doy = pd.to_datetime(meta["target_date"]).dt.dayofyear.values
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["spatial_group"] = (
        np.floor(out["lat"] / cell_deg).astype(int).astype(str) + "_" +
        np.floor(out["lon"] / cell_deg).astype(int).astype(str)
    )
    return out


def select_features(df: pd.DataFrame, use_coords: bool, use_doy: bool) -> list:
    drop = set(META_COLS)
    if not use_coords:
        drop |= set(COORD_COLS)
    if not use_doy:
        drop |= set(DOY_COLS)
    return [c for c in df.columns if c not in drop]


def cv_compare(df: pd.DataFrame, n_splits: int = 5, iterations: int = 400) -> pd.DataFrame:
    """
    Honest comparison of feature sets x grouping schemes with ALL folds (mean +- sd).
    Fixed number of trees (no early stopping on the validation fold -> no selection bias).
    group = scene_id : old scheme, neighbouring scenes may sit in train and val.
    group = cell     : 0.5-degree cells, neighbours stay together (needed when lat/lon are features).
    """
    configs = [
        ("no coords", False, False, "scene_id"),
        ("no coords", False, False, "spatial_group"),
        ("+ lat/lon", True, False, "scene_id"),
        ("+ lat/lon", True, False, "spatial_group"),
        ("+ lat/lon + doy", True, True, "spatial_group"),
    ]
    y = df["is_fire"].astype(int)
    rows = []
    for name, use_c, use_d, gcol in configs:
        cols = select_features(df, use_c, use_d)
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        roc, pr = [], []
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=df[gcol]):
            pos = float((y.iloc[tr] == 0).sum() / max((y.iloc[tr] == 1).sum(), 1))
            m = CatBoostClassifier(iterations=iterations, learning_rate=0.05, depth=6,
                                   scale_pos_weight=pos, random_seed=42, verbose=0)
            m.fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            roc.append(roc_auc_score(y.iloc[va], p))
            pr.append(average_precision_score(y.iloc[va], p))
        rows.append({"features": name, "groups": gcol, "n_feat": len(cols),
                     "ROC-AUC": f"{np.mean(roc):.3f} ± {np.std(roc):.3f}",
                     "PR-AUC": f"{np.mean(pr):.3f} ± {np.std(pr):.3f}"})
        print(f"[CV] {name:16s} groups={gcol:14s} ROC {np.mean(roc):.3f}±{np.std(roc):.3f}  "
              f"PR {np.mean(pr):.3f}±{np.std(pr):.3f}", flush=True)
    return pd.DataFrame(rows)


def train_catboost(df: pd.DataFrame, output_model_path: str, use_coords: bool = False,
                   use_doy: bool = False, group_col: str = "scene_id") -> None:
    feature_cols = select_features(df, use_coords, use_doy)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["is_fire"].astype(int)
    groups = df[group_col]

    print("\n" + "=" * 65)
    print(f"[INFO] Feature Matrix Shape:      {X.shape}")
    print(f"[INFO] Total Input Features:      {len(feature_cols)}  (coords={use_coords}, doy={use_doy}, groups={group_col})")

    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    print(f"[INFO] Train Split: {len(X_train):6d} patches (Fire=1: {y_train.sum()})")
    print(f"[INFO] Val Split:   {len(X_val):6d} patches (Fire=1: {y_val.sum()})")

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos = float(neg_count / max(pos_count, 1))
    print(f"[INFO] Computed scale_pos_weight: {scale_pos:.3f}")

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos,
        random_seed=42,
        early_stopping_rounds=60,
        verbose=100,
    )

    print("\n[INFO] Fitting CatBoost Classifier...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds_proba = model.predict_proba(X_val)[:, 1]
    preds_binary = (preds_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, preds_proba)
    pr_auc = average_precision_score(y_val, preds_proba)

    print("\n" + "=" * 65)
    print(f"VALIDATION RESULTS (GROUP-KFOLD BY {group_col.upper()}, FOLD 0):")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nClassification Report (Threshold = 0.5):")
    print(classification_report(y_val, preds_binary, digits=4))
    print("=" * 65)

    importance = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values(by="importance", ascending=False)

    print("\nTOP-20 MOST IMPORTANT FEATURES:")
    print(imp_df.head(20).to_string(index=False))

    model.save_model(output_model_path)
    print(f"\n[SUCCESS] Saved trained model to: {output_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline on wildfire HDF5 dataset.")
    parser.add_argument("--h5", default="data/processed/dataset.h5", help="Path to HDF5 dataset.")
    parser.add_argument("--output", default="catboost_baseline.cbm", help="Output model path.")
    parser.add_argument("--cache_parquet", default="data/processed/tabular_features.parquet", help="Feature cache path.")
    parser.add_argument("--use_coords", action="store_true", help="Use lat/lon as features in the final model.")
    parser.add_argument("--use_doy", action="store_true", help="Use day-of-year (sin/cos) as features in the final model.")
    parser.add_argument("--group_by", default="auto", choices=["auto", "scene_id", "spatial_group"],
                        help="CV grouping for the final model. auto = spatial_group if coords are used, else scene_id.")
    parser.add_argument("--compare", action="store_true",
                        help="Run the 5-fold comparison table (no coords / + lat,lon / + doy; scene vs 0.5-degree groups).")
    args = parser.parse_args()

    cache_path = Path(args.cache_parquet)
    if cache_path.exists():
        print(f"[INFO] Loading cached tabular features from: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        df = extract_tabular_dataset(args.h5)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[SUCCESS] Cached extracted features to: {cache_path}")

    # Old caches have no coordinates: add them from metadata only (seconds, no raster re-read)
    if "lat" not in df.columns or "spatial_group" not in df.columns:
        df = add_meta_features(df, args.h5)
        df.to_parquet(cache_path, index=False)
        print("[INFO] Added lat/lon/doy/spatial_group to the cache.")

    if args.compare:
        table = cv_compare(df)
        print("\n" + table.to_string(index=False))

    group_col = args.group_by
    if group_col == "auto":
        group_col = "spatial_group" if args.use_coords else "scene_id"
    train_catboost(df, args.output, use_coords=args.use_coords, use_doy=args.use_doy, group_col=group_col)


if __name__ == "__main__":
    main()