"""Risk map and visual multi-model comparison over the Iberia scene grid.

Usage examples:
  python make_map.py --date 2025-08-12 --points metadata.csv --oof oof_predictions.csv
  python make_map.py --date 2025-08-12 --points metadata.csv --models tab_clim_full,nn_ts,tab_hist --compare tab_clim_full,nn_ts
  python make_map.py --date 2025-08-12 --points metadata.csv --collect-2d-limit 20 --models tab_ssl_full,tab_full,tab_clim_full --compare tab_ssl_full,tab_clim_full
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import live
from config_and_features import (
    CLIM_COL_NAMES,
    CONFIG,
    CONTEXT_1D,
    HIST_COL_NAMES,
    HIST_WINDOWS,
    IBERIA_BBOX,
    RADII,
    R_EARTH,
    TAB_1D_COL_NAMES,
    build_seq_and_static,
    circ,
    climatology_many,
    history_many,
    one_d_features,
    to_xyz,
)
from data_collectors import (
    Scene2DCollector,
    fetch_firms_days,
    fetch_population_points,
    fetch_weather_cells,
)
from mini_dataset_store import MiniDatasetStore
from model_registry import COLORS, LEVELS, MODEL_CATALOG, ModelRegistry


def load_points(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    la = next(c for c in ("lat", "latitude") if c in df.columns)
    lo = next(c for c in ("lon", "longitude") if c in df.columns)
    pts = df[[la, lo]].round(5).drop_duplicates().reset_index(drop=True)
    pts.columns = ["lat", "lon"]
    return pts, df


def build_grid_features(
    pts: pd.DataFrame,
    day: date,
    meta_df: Optional[pd.DataFrame] = None,
    collect_2d_limit: int = 0,
    collect_2d_strategy: str = "top_risk",
    store: Optional[MiniDatasetStore] = None,
    reg: Optional[ModelRegistry] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    store = store or MiniDatasetStore(CONFIG.get("MINI_DATASET_DIR", "mini_dataset"))
    reg = reg or ModelRegistry(CONFIG.get("MODEL_DIR", "models"))

    wd = day - timedelta(days=1)
    gap = int(CONFIG["WEATHER_GAP_DAYS"])
    H = int(CONFIG["HISTORY_DAYS"])

    if progress_cb:
        progress_cb(0.02, f"[Step 1/5] Fetching Open-Meteo ERA5 weather for {wd} across {len(pts)} points...")
    print(f"[1/5] Fetching weather for {wd} across {len(pts)} points...")
    w_list, n_cells = fetch_weather_cells(pts, day, gap, progress_cb=progress_cb)
    print(f"      Fetched {n_cells} unique 0.25 deg cells.")

    if progress_cb:
        progress_cb(0.36, "[Step 2/5] Loading population features (10 km and 50 km)...")
    print("[2/5] Loading population features...")
    pop = fetch_population_points(pts, cache=CONFIG["POPULATION_CACHE_CSV"], metadata_df=meta_df, progress_cb=progress_cb)

    scenes_2d: List[Optional[Dict[str, Any]]] = [None] * len(pts)
    s2_dates: List[Optional[str]] = [None] * len(pts)
    for i, r in pts.iterrows():
        cp = store.has_scene_npz(r.lat, r.lon, day)
        if cp is not None:
            try:
                scenes_2d[i] = store.load_scene_npz(cp)
            except Exception:
                pass

    raw_1d_list, tab_1d_list = [], []
    for i, w in enumerate(w_list):
        r1d, t1d = one_d_features(
            w,
            float(pop.population_10km_log1p.iloc[i]),
            float(pop.population_50km_log1p.iloc[i]),
            wd,
            s2_date=s2_dates[i],
        )
        raw_1d_list.append(r1d)
        tab_1d_list.append(t1d)

    RAW_1D = np.vstack(raw_1d_list).astype(np.float32)
    TAB_1D = np.vstack(tab_1d_list).astype(np.float32)

    if progress_cb:
        progress_cb(0.46, f"[Step 3/5] Fetching {H}-day FIRMS history ({day - timedelta(days=H)} to {wd})...")
    print(f"[3/5] Fetching {H}-day FIRMS history ({day - timedelta(days=H)} to {wd})...")
    fire_df, covered = fetch_firms_days(day - timedelta(days=H), wd, bbox=IBERIA_BBOX, progress_cb=progress_cb)
    if progress_cb:
        progress_cb(0.60, f"[Step 3/5] Aggregating spatial fire history across 4 radii (5, 10, 25, 50 km) for {len(pts)} points...")
    pts_xyz = to_xyz(pts.lat.values, pts.lon.values)
    HIST_TAB, HC = history_many(pts_xyz, fire_df, covered, day, RADII, H=H)
    print(f"      {len(fire_df)} VIIRS detections across {len(covered)}/{H} covered days.")

    if progress_cb:
        progress_cb(0.66, "[Step 4/5] Computing multi-year historical day-of-year climatology (KDTree)...")
    print("[4/5] Computing historical day-of-year climatology...")
    CLIM_FEATS = climatology_many(
        pts_xyz,
        day,
        RADII,
        int(CONFIG["CLIM_WINDOW_DAYS"]),
        CONFIG["HISTORICAL_FIRMS_CSV"],
        CONFIG["HISTORICAL_COVERED_CSV"],
    )

    SEQ, STATIC_1D = build_seq_and_static(HC, TAB_1D, CLIM_FEATS, H=H)

    if collect_2d_limit > 0:
        try:
            prelim = reg.predict_model("tab_clim_full", TAB_1D, HIST_TAB, CLIM_FEATS)["p"]
        except Exception:
            prelim = reg.predict_model("persist", TAB_1D, HIST_TAB, CLIM_FEATS)["p"]

        uncollected = [i for i in range(len(pts)) if scenes_2d[i] is None]
        if collect_2d_strategy == "random":
            rng = np.random.default_rng(42)
            chosen = list(rng.choice(uncollected, size=min(collect_2d_limit, len(uncollected)), replace=False))
        else:
            chosen = sorted(uncollected, key=lambda idx: float(prelim[idx]), reverse=True)[:collect_2d_limit]

        collector = Scene2DCollector()
        for rank_i, idx in enumerate(chosen):
            la, lo = float(pts.lat.iloc[idx]), float(pts.lon.iloc[idx])
            if progress_cb:
                progress_cb(
                    0.70 + 0.15 * (rank_i / max(len(chosen), 1)),
                    f"Collecting 2D satellite scene {rank_i + 1}/{len(chosen)}: ({la:.4f}, {lo:.4f})...",
                )
            print(f"      [2D {rank_i + 1}/{len(chosen)}] Collecting 22-ch patch at ({la:.4f}, {lo:.4f})...")
            try:
                sc = collector.collect_scene(
                    lat=la,
                    lon=lo,
                    target_day=day,
                    weather_dict=w_list[idx],
                    pop_10_50=(float(pop.population_10km_log1p.iloc[idx]), float(pop.population_50km_log1p.iloc[idx])),
                )
                scenes_2d[idx] = sc
                store.save_scene_npz(sc)
                if sc.get("s2_Source_Date") and sc["s2_Source_Date"] != "None":
                    r1d_new, t1d_new = one_d_features(
                        w_list[idx],
                        float(pop.population_10km_log1p.iloc[idx]),
                        float(pop.population_50km_log1p.iloc[idx]),
                        wd,
                        s2_date=sc["s2_Source_Date"],
                    )
                    RAW_1D[idx] = r1d_new
                    TAB_1D[idx] = t1d_new
            except Exception as e:
                print(f"      [2D warning] ({la:.4f}, {lo:.4f}) failed: {e}")

    FT_2D, EMB32_2D, has_2d_mask = reg.compute_2d_embeddings_and_stats(scenes_2d)

    feat_df = pd.concat(
        [
            pts.reset_index(drop=True),
            pd.DataFrame({"date": str(day), "has_2d_scene": has_2d_mask}),
            pd.DataFrame(TAB_1D, columns=TAB_1D_COL_NAMES),
            pd.DataFrame(HIST_TAB, columns=HIST_COL_NAMES),
            pd.DataFrame(CLIM_FEATS, columns=CLIM_COL_NAMES),
        ],
        axis=1,
    )
    store.save_daily_features(day, feat_df)

    return {
        "pts": pts,
        "day": day,
        "RAW_1D": RAW_1D,
        "TAB_1D": TAB_1D,
        "HIST_TAB": HIST_TAB,
        "CLIM_FEATS": CLIM_FEATS,
        "SEQ": SEQ,
        "STATIC_1D": STATIC_1D,
        "FT_2D": FT_2D,
        "EMB32_2D": EMB32_2D,
        "has_2d_mask": has_2d_mask,
        "n_weather_cells": n_cells,
        "n_firms_detections": len(fire_df),
        "n_covered_days": len(covered),
        "n_2d_scenes": int(has_2d_mask.sum()),
    }


def draw_single_map(
    pts: pd.DataFrame,
    model_out: Dict[str, Any],
    day: date,
    png_path: str | Path,
    has_2d_mask: Optional[np.ndarray] = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lev = model_out["level_idx"]
    t = model_out["thresholds"]
    mu50 = model_out["mu"][50]
    m_name = model_out["model"]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), dpi=150)

    for i, name in enumerate(LEVELS):
        m = lev == i
        ax[0].scatter(
            pts.lon[m],
            pts.lat[m],
            c=COLORS[i],
            s=14,
            label=f"{name} ({m.sum()} scenes, {100 * m.mean():.0f}%)",
            edgecolors="none",
        )
    if has_2d_mask is not None and has_2d_mask.any():
        ax[0].scatter(
            pts.lon[has_2d_mask],
            pts.lat[has_2d_mask],
            facecolors="none",
            edgecolors="#00e5ff",
            s=42,
            linewidths=1.2,
            label=f"2D scene ({int(has_2d_mask.sum())})",
        )
    ax[0].set_title(
        f"[{m_name}] Wildfire risk level on {day}\n"
        f"(OOF thresholds: Low < {t[0]:.2f} <= Med < {t[1]:.2f} <= High)"
    )
    ax[0].set_xlabel("Longitude")
    ax[0].set_ylabel("Latitude")
    ax[0].legend(loc="lower left", frameon=True, fontsize=8.5)
    ax[0].grid(alpha=0.25)

    vmax = max(1.0, float(np.quantile(mu50, 0.98))) if len(mu50) else 1.0
    sc = ax[1].scatter(pts.lon, pts.lat, c=mu50, s=14, cmap="YlOrRd", vmin=0, vmax=vmax, edgecolors="none")
    plt.colorbar(sc, ax=ax[1], label="E[VIIRS detections within 50 km]")
    ax[1].set_title(f"[{m_name}] Expected detections within 50 km on {day}")
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    ax[1].grid(alpha=0.25)

    plt.tight_layout()
    png_path = Path(png_path)
    plt.savefig(png_path)
    plt.close(fig)
    return png_path


def draw_comparison_map(
    pts: pd.DataFrame,
    out_a: Dict[str, Any],
    out_b: Dict[str, Any],
    day: date,
    png_path: str | Path,
    has_2d_mask: Optional[np.ndarray] = None,
) -> Tuple[Path, Dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pa, pb = out_a["p"], out_b["p"]
    la, lb = out_a["level_idx"], out_b["level_idx"]
    ta, tb = out_a["thresholds"], out_b["thresholds"]
    na, nb = out_a["model"], out_b["model"]
    dp = pa - pb
    agree = float((la == lb).mean())
    corr = float(np.corrcoef(pa, pb)[0, 1]) if (np.std(pa) > 1e-8 and np.std(pb) > 1e-8) else 1.0

    fig, ax = plt.subplots(1, 3, figsize=(20, 6.2), dpi=150)

    for panel_idx, (m_name, lev, t, p_arr) in enumerate([(na, la, ta, pa), (nb, lb, tb, pb)]):
        for i, name in enumerate(LEVELS):
            m = lev == i
            ax[panel_idx].scatter(
                pts.lon[m],
                pts.lat[m],
                c=COLORS[i],
                s=13,
                label=f"{name}: {m.sum()} ({100 * m.mean():.0f}%)",
                edgecolors="none",
            )
        if has_2d_mask is not None and has_2d_mask.any():
            ax[panel_idx].scatter(
                pts.lon[has_2d_mask],
                pts.lat[has_2d_mask],
                facecolors="none",
                edgecolors="#00e5ff",
                s=40,
                linewidths=1.1,
                label=f"2D scene ({int(has_2d_mask.sum())})",
            )
        ax[panel_idx].set_title(
            f"Model A: {m_name}\nThresholds: {t[0]:.2f} / {t[1]:.2f} | Mean P={p_arr.mean():.3f}"
            if panel_idx == 0
            else f"Model B: {m_name}\nThresholds: {t[0]:.2f} / {t[1]:.2f} | Mean P={p_arr.mean():.3f}"
        )
        ax[panel_idx].set_xlabel("Longitude")
        ax[panel_idx].set_ylabel("Latitude")
        ax[panel_idx].legend(loc="lower left", frameon=True, fontsize=8)
        ax[panel_idx].grid(alpha=0.25)

    lim = max(0.15, float(np.quantile(np.abs(dp), 0.98))) if len(dp) else 0.25
    sc = ax[2].scatter(pts.lon, pts.lat, c=dp, s=13, cmap="coolwarm", vmin=-lim, vmax=lim, edgecolors="none")
    plt.colorbar(sc, ax=ax[2], label=f"Delta P = P({na}) - P({nb})")
    ax[2].set_title(
        f"Probability Difference ({na} - {nb})\n"
        f"Level agreement: {100 * agree:.1f}% | Correlation r = {corr:.3f}"
    )
    ax[2].set_xlabel("Longitude")
    ax[2].set_ylabel("Latitude")
    ax[2].grid(alpha=0.25)

    plt.tight_layout()
    png_path = Path(png_path)
    plt.savefig(png_path)
    plt.close(fig)

    stats = {
        "model_a": na,
        "model_b": nb,
        "level_agreement_pct": round(100.0 * agree, 2),
        "pearson_r": round(corr, 4),
        "mean_p_a": round(float(pa.mean()), 4),
        "mean_p_b": round(float(pb.mean()), 4),
        "mean_abs_diff": round(float(np.abs(dp).mean()), 4),
        "max_abs_diff": round(float(np.abs(dp).max()), 4),
    }
    return png_path, stats


def main():
    ap = argparse.ArgumentParser(description="PreemptFire daily Iberia risk map and model comparison")
    ap.add_argument("--date", required=True, help="Scored day YYYY-MM-DD (uses weather and FIRMS up to day - 1)")
    ap.add_argument("--points", default="metadata.csv", help="CSV with lat/latitude and lon/longitude columns")
    ap.add_argument("--oof", default="oof_predictions.csv", help="Out-of-fold predictions CSV for fixed thresholds")
    ap.add_argument("--shares", default="0.60,0.30", help="Low,Medium scene shares on OOF (rest is High)")
    ap.add_argument("--models", default="tab_clim_full,persist,clim", help="Comma-separated models to score (or 'all')")
    ap.add_argument("--compare", default="", help="Optional pair of models to compare side-by-side, e.g. 'tab_clim_full,nn_ts'")
    ap.add_argument("--collect-2d-limit", type=int, default=0, help="Number of grid points to collect live 2D satellite scenes for")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    day = datetime.strptime(args.date, "%Y-%m-%d").date()
    out_csv = args.out_csv or f"risk_map_{day}.csv"
    out_png = args.out_png or f"risk_map_{day}.png"

    pts, meta_df = load_points(args.points)
    print(f"Loaded {len(pts)} unique Iberia locations from {args.points}")

    store = MiniDatasetStore(CONFIG.get("MINI_DATASET_DIR", "mini_dataset"))
    reg = ModelRegistry(model_dir=CONFIG.get("MODEL_DIR", "models"), oof_path=args.oof)

    grid_data = build_grid_features(
        pts=pts,
        day=day,
        meta_df=meta_df,
        collect_2d_limit=args.collect_2d_limit,
        store=store,
        reg=reg,
    )

    if args.models.strip().lower() == "all":
        model_names = reg.available_models(include_2d=True)
    else:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    out_df = pts.copy()
    out_df["date"] = str(day)
    out_df["has_2d_scene"] = grid_data["has_2d_mask"]
    out_df["recent_fires_7d_50km"] = grid_data["HIST_TAB"][:, 2 * 4 + 3].astype(int)
    out_df["recent_fires_30d_50km"] = grid_data["HIST_TAB"][:, 4 * 4 + 3].astype(int)

    model_outputs: Dict[str, Dict[str, Any]] = {}
    for m_name in model_names:
        try:
            res = reg.predict_model(
                model_name=m_name,
                TAB_1D=grid_data["TAB_1D"],
                HIST_TAB=grid_data["HIST_TAB"],
                CLIM_FEATS=grid_data["CLIM_FEATS"],
                SEQ=grid_data["SEQ"],
                STATIC_1D=grid_data["STATIC_1D"],
                pts_lat_lon=pts[["lat", "lon"]].values,
                FT_2D=grid_data["FT_2D"],
                EMB32_2D=grid_data["EMB32_2D"],
                has_2d_mask=grid_data["has_2d_mask"],
            )
            model_outputs[m_name] = res
            out_df[f"P_fire_50km_{m_name}"] = np.round(res["p"], 4)
            out_df[f"risk_level_{m_name}"] = res["levels"]
            out_df[f"expected_50km_{m_name}"] = np.round(res["mu"][50], 2)
            out_df[f"upper90_50km_{m_name}"] = res["upper90"][50]
        except Exception as e:
            print(f"  [skip {m_name}] {e}")

    if not model_outputs:
        raise RuntimeError("No models could be evaluated. Check the models/ directory.")

    primary_name = next(iter(model_outputs.keys()))
    prim = model_outputs[primary_name]
    out_df["P_fire_50km"] = np.round(prim["p"], 4)
    out_df["risk_level"] = prim["levels"]
    for R in RADII:
        out_df[f"expected_{R}km"] = np.round(prim["mu"][R], 2)
        out_df[f"upper_90pct_{R}km"] = prim["upper90"][R]

    out_df.to_csv(out_csv, index=False)
    store.save_daily_predictions(day, out_df)

    draw_single_map(pts, prim, day, out_png, has_2d_mask=grid_data["has_2d_mask"])
    draw_single_map(pts, prim, day, store.maps_dir / f"risk_map_{primary_name}_{day}.png", has_2d_mask=grid_data["has_2d_mask"])

    cmp_info = None
    if args.compare:
        pair = [x.strip() for x in args.compare.split(",") if x.strip()]
        if len(pair) == 2 and pair[0] in model_outputs and pair[1] in model_outputs:
            cmp_png = f"compare_{pair[0]}_vs_{pair[1]}_{day}.png"
            _, cmp_info = draw_comparison_map(
                pts,
                model_outputs[pair[0]],
                model_outputs[pair[1]],
                day,
                cmp_png,
                has_2d_mask=grid_data["has_2d_mask"],
            )
            draw_comparison_map(
                pts,
                model_outputs[pair[0]],
                model_outputs[pair[1]],
                day,
                store.maps_dir / cmp_png,
                has_2d_mask=grid_data["has_2d_mask"],
            )

    summary = {
        "date": str(day),
        "n_points": len(pts),
        "primary_model": primary_name,
        "models_scored": list(model_outputs.keys()),
        "n_2d_scenes": grid_data["n_2d_scenes"],
        "counts_today_primary": {k: int((prim["levels"] == k).sum()) for k in LEVELS},
        "comparison": cmp_info,
        "out_csv": out_csv,
        "out_png": out_png,
        "mini_dataset_dir": str(store.root),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
