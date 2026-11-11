"""Single-location live inference & 2D/1D scene collector for PreemptFire.

Drop-in replacement for Claude's `live.py` that:
1. Uses exact repository (`approach_b`) + notebook (`preemptfire-v5_(2).ipynb`) feature formulas.
2. Supports BOTH 1D models (`tab_clim_full`, `tab_hist`, `tab_1d`, `nn_ts`, `coords`, `persist`, `clim`)
   AND 2D satellite image models (`tab_ssl_full`, `tab_full`, `stack_all`) via `Scene2DCollector`.
3. Automatically saves collected 2D/1D scenes into `mini_dataset/scenes/` (`.npz` + `metadata.csv`).

Usage:
  # Fast 1D + History + Climatology across all non-image models:
  python live.py --lat 40.0 --lon -4.0 --date 2025-08-12

  # Full 2D + 1D collection (Sentinel-2 + Sentinel-1 + DEM + OSM + Soil Moisture) + ALL models (including tab_ssl_full & tab_full):
  python live.py --lat 40.0 --lon -4.0 --date 2025-08-12 --collect-2d --models all
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config_and_features import (
    CONFIG,
    CONTEXT_1D,
    HIST_WINDOWS,
    RADII,
    R_EARTH,
    build_seq_and_static,
    circ,
    climatology_many,
    dist_km,
    history_many,
    one_d_features,
    to_xyz,
)
from data_collectors import (
    Scene2DCollector,
    fetch_firms_days,
    fetch_population_single,
    fetch_weather,
)
from mini_dataset_store import MiniDatasetStore
from model_registry import MODEL_CATALOG, ModelRegistry


def score_single_location(
    lat: float,
    lon: float,
    target_day_str: str,
    models: Optional[List[str]] = None,
    collect_2d: bool = False,
    save_to_mini_dataset: bool = True,
    model_dir: Optional[str] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    """Collects live data for `(lat, lon, target_day_str)`, optionally collects/loads the 22-channel
    2D satellite patch, saves to `mini_dataset/`, and scores all requested models.
    """
    day = datetime.strptime(str(target_day_str)[:10], "%Y-%m-%d").date()
    wd = day - timedelta(days=1)
    gap = int(CONFIG["WEATHER_GAP_DAYS"])
    t0 = wd - timedelta(days=gap)

    store = MiniDatasetStore(CONFIG.get("MINI_DATASET_DIR", "mini_dataset"))
    reg = ModelRegistry(model_dir=model_dir or CONFIG.get("MODEL_DIR", "models"))

    if progress_cb:
        progress_cb(0.08, f"[Step 1/6] Fetching 60-day Open-Meteo ERA5 weather for ({lat:.4f}, {lon:.4f}) up to {wd}...")
    w = fetch_weather(lat, lon, wd, t0)

    if progress_cb:
        progress_cb(0.16, f"[Step 2/6] Fetching population density (10km / 50km) for ({lat:.4f}, {lon:.4f})...")
    p10, p50 = fetch_population_single(lat, lon)

    scene_2d: Optional[Dict[str, Any]] = None
    scene_2d_error: Optional[str] = None
    cached_npz = store.has_scene_npz(lat, lon, day)
    if cached_npz is not None:
        if progress_cb:
            progress_cb(0.45, f"[Step 3/6] Loaded cached 22-channel 2D satellite patch from {cached_npz.name}...")
        scene_2d = store.load_scene_npz(cached_npz)
    elif collect_2d:
        try:
            collector = Scene2DCollector()
            scene_2d = collector.collect_scene(
                lat=lat,
                lon=lon,
                target_day=day,
                weather_dict=w,
                pop_10_50=(p10, p50),
                progress_cb=progress_cb,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            scene_2d_error = f"{type(e).__name__}: {e}"
            print(f"  [2D Collector] Failed to collect 2D patch: {scene_2d_error}")

    s2_date = scene_2d.get("s2_Source_Date") if scene_2d else None
    raw_1d, tab_1d = one_d_features(w, p10, p50, wd, s2_date=s2_date)
    TAB_1D = tab_1d[None, :]  # (1, 39)

    H = int(CONFIG["HISTORY_DAYS"])
    if progress_cb:
        progress_cb(0.68, f"[Step 4/6] Fetching {H}-day NASA FIRMS fire detections ({day - timedelta(days=H)} to {wd})...")
    fire_df, covered = fetch_firms_days(lat, lon, day - timedelta(days=H), wd)
    pts_xyz = to_xyz([lat], [lon])
    HIST_TAB, HC = history_many(pts_xyz, fire_df, covered, day, RADII, H=H)

    if progress_cb:
        progress_cb(0.78, "[Step 5/6] Computing historical DOY fire climatology...")
    CLIM_FEATS = climatology_many(
        pts_xyz,
        day,
        RADII,
        int(CONFIG["CLIM_WINDOW_DAYS"]),
        CONFIG["HISTORICAL_FIRMS_CSV"],
        CONFIG["HISTORICAL_COVERED_CSV"],
    )

    SEQ, STATIC_1D = build_seq_and_static(HC, TAB_1D, CLIM_FEATS, H=H)

    if progress_cb:
        progress_cb(0.88, "[Step 6/6] Extracting 2D embeddings (SimCLR + FT) and running model inference...")
    FT_2D, EMB32_2D, has_2d_mask = reg.compute_2d_embeddings_and_stats([scene_2d])

    # Determine which models to run
    avail = reg.available_models(include_2d=bool(has_2d_mask[0]))
    if not models or models == ["all"] or "all" in models:
        selected_models = avail
    else:
        selected_models = [m for m in models if m in MODEL_CATALOG]

    predictions: Dict[str, Any] = {}
    for m in selected_models:
        try:
            out = reg.predict_model(
                model_name=m,
                TAB_1D=TAB_1D,
                HIST_TAB=HIST_TAB,
                CLIM_FEATS=CLIM_FEATS,
                SEQ=SEQ,
                STATIC_1D=STATIC_1D,
                pts_lat_lon=np.array([[lat, lon]], dtype=np.float32),
                FT_2D=FT_2D,
                EMB32_2D=EMB32_2D,
                has_2d_mask=has_2d_mask,
            )
            predictions[m] = {
                "label": out["label"],
                "P_fire_50km": round(float(out["p"][0]), 4),
                "risk_level": str(out["levels"][0]),
                "thresholds_low_med_high": [round(float(x), 4) for x in out["thresholds"]],
                "expected_count": {f"{R}km": round(float(out["mu"][R][0]), 2) for R in RADII},
                "upper_90pct_count": {f"{R}km": int(out["upper90"][R][0]) for R in RADII},
                "used_2d_imagery": bool(MODEL_CATALOG[m]["requires_2d"] and has_2d_mask[0]),
            }
        except Exception as e:
            predictions[m] = {"error": str(e)}

    # Save 2D scene + 1D vector + predictions into mini_dataset
    saved_npz_path = None
    if save_to_mini_dataset and scene_2d is not None:
        scene_to_save = dict(scene_2d)
        scene_to_save["latitude"] = lat
        scene_to_save["longitude"] = lon
        scene_to_save["date"] = str(day)
        scene_to_save["X_1d"] = raw_1d
        flat_preds = {
            f"p_{m}": v["P_fire_50km"]
            for m, v in predictions.items()
            if isinstance(v, dict) and "P_fire_50km" in v
        }
        scene_to_save["predictions"] = flat_preds
        saved_npz_path = str(store.save_scene_npz(scene_to_save))

    return {
        "lat": lat,
        "lon": lon,
        "date": str(day),
        "has_2d_scene": bool(has_2d_mask[0]),
        "scene_2d_error": scene_2d_error,
        "s2_source_date": s2_date,
        "saved_npz_path": saved_npz_path,
        "recent_fires_7d_50km": int(HIST_TAB[0, 2 * 4 + 3]),
        "recent_fires_30d_50km": int(HIST_TAB[0, 4 * 4 + 3]),
        "predictions": predictions,
        "scene_2d": scene_2d,
    }


def main():
    ap = argparse.ArgumentParser(description="Live single-location wildfire risk inference (1D + 2D satellite models)")
    ap.add_argument("--lat", type=float, required=True, help="Latitude (e.g. 40.0)")
    ap.add_argument("--lon", type=float, required=True, help="Longitude (e.g. -4.0)")
    ap.add_argument("--date", type=str, required=True, help="Scored date YYYY-MM-DD (e.g. 2025-08-12)")
    ap.add_argument("--models", type=str, default="all", help="Comma-separated model names or 'all'")
    ap.add_argument("--collect-2d", action="store_true", help="Collect 22-channel 2D satellite patch (Sentinel-2, S1, DEM, OSM, ERA5) for tab_ssl_full & tab_full")
    ap.add_argument("--model-dir", type=str, default=CONFIG["MODEL_DIR"])
    args = ap.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    res = score_single_location(
        lat=args.lat,
        lon=args.lon,
        target_day_str=args.date,
        models=model_list,
        collect_2d=args.collect_2d,
        save_to_mini_dataset=True,
        model_dir=args.model_dir,
    )
    # Remove raw numpy arrays before JSON printing
    res.pop("scene_2d", None)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
