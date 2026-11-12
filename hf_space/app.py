"""Gradio web application for PreemptFire on Hugging Face Spaces."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_and_features import CI, CONFIG, RADII, build_2d_np, get_firms_keys
from live import score_single_location
from make_map import (
    build_grid_features,
    draw_comparison_map,
    draw_single_map,
    load_points,
)
from mini_dataset_store import MiniDatasetStore
from model_registry import COLORS, LEVELS, MODEL_CATALOG, ModelRegistry
from post_hoc_evaluation import (
    compute_actual_counts_per_point,
    draw_ground_truth_comparison_map,
    evaluate_levels_against_actual,
    evaluate_local_radius_around_point,
    evaluate_tier_reliability_table,
    evaluate_vs_fwi_benchmark,
    fetch_post_hoc_firms,
)

try:
    import spaces
    GPU = spaces.GPU
except Exception:
    def GPU(func=None, **kwargs):
        if func is not None:
            return func
        def dec(f):
            return f
        return dec

_GRID_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_registry_and_store() -> Tuple[ModelRegistry, MiniDatasetStore]:
    store = MiniDatasetStore(CONFIG.get("MINI_DATASET_DIR", "mini_dataset"))
    reg = ModelRegistry(model_dir=CONFIG.get("MODEL_DIR", "models"))
    return reg, store


def _find_points_csv() -> str:
    for c in ("metadata.csv", "data/metadata.csv", "../metadata.csv"):
        if Path(c).exists():
            return c
    return "metadata.csv"


def _model_choices() -> List[Tuple[str, str]]:
    reg, _ = _get_registry_and_store()
    rep = reg.status_report()
    choices = []
    for k, info in MODEL_CATALOG.items():
        avail = rep.get(k, {}).get("available", False)
        status_tag = "ready" if avail else "weights missing"
        choices.append((f"[{status_tag}] {k}: {info['label']}", k))
    return choices


def _parse_day(date_str: str):
    s = str(date_str or "").strip()[:10]
    if not s:
        s = "2025-08-12"
    return datetime.strptime(s, "%Y-%m-%d").date()


@GPU
def _gpu_predict_torch(
    reg: ModelRegistry,
    model_name: str,
    tab_1d: np.ndarray,
    hist_tab: np.ndarray,
    clim_feats: np.ndarray,
    seq: Optional[np.ndarray],
    static_1d: Optional[np.ndarray],
    pts_ll: Optional[np.ndarray],
    ft_2d: Optional[np.ndarray],
    emb32_2d: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    return reg.predict_model(
        model_name=model_name,
        TAB_1D=tab_1d,
        HIST_TAB=hist_tab,
        CLIM_FEATS=clim_feats,
        SEQ=seq,
        STATIC_1D=static_1d,
        pts_lat_lon=pts_ll,
        FT_2D=ft_2d,
        EMB32_2D=emb32_2d,
        has_2d_mask=mask,
    )


def gpu_predict(
    reg: ModelRegistry,
    model_name: str,
    tab_1d: np.ndarray,
    hist_tab: np.ndarray,
    clim_feats: np.ndarray,
    seq: Optional[np.ndarray],
    static_1d: Optional[np.ndarray],
    pts_ll: Optional[np.ndarray],
    ft_2d: Optional[np.ndarray],
    emb32_2d: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    if model_name in ("nn_ts", "stack_all"):
        return _gpu_predict_torch(
            reg=reg,
            model_name=model_name,
            tab_1d=tab_1d,
            hist_tab=hist_tab,
            clim_feats=clim_feats,
            seq=seq,
            static_1d=static_1d,
            pts_ll=pts_ll,
            ft_2d=ft_2d,
            emb32_2d=emb32_2d,
            mask=mask,
        )
    return reg.predict_model(
        model_name=model_name,
        TAB_1D=tab_1d,
        HIST_TAB=hist_tab,
        CLIM_FEATS=clim_feats,
        SEQ=seq,
        STATIC_1D=static_1d,
        pts_lat_lon=pts_ll,
        FT_2D=ft_2d,
        EMB32_2D=emb32_2d,
        has_2d_mask=mask,
    )


def run_iberia_map_and_compare(
    date_str: str,
    model_a: str,
    model_b: str,
    also_score_all: bool,
    collect_2d_limit: int,
    collect_2d_strategy: str,
    force_refresh: bool,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        day = _parse_day(date_str)
        reg, store = _get_registry_and_store()
        pts_path = _find_points_csv()
        pts, meta_df = load_points(pts_path)

        cache_key = f"{day}_{len(pts)}"
        if force_refresh or cache_key not in _GRID_CACHE or int(collect_2d_limit) > 0:
            grid_data = build_grid_features(
                pts=pts,
                day=day,
                meta_df=meta_df,
                collect_2d_limit=int(collect_2d_limit),
                collect_2d_strategy=collect_2d_strategy,
                store=store,
                reg=reg,
                progress_cb=lambda frac, desc: progress(frac, desc=desc),
            )
            _GRID_CACHE[cache_key] = grid_data
        else:
            progress(0.35, desc=f"Loading cached features for {day}...")
            grid_data = _GRID_CACHE[cache_key]

        progress(0.82, desc="Evaluating models and rendering maps...")
        models_to_run = []
        for m in (model_a, model_b):
            if m and m not in models_to_run:
                models_to_run.append(m)
        if also_score_all:
            for m in reg.available_models(include_2d=True):
                if m not in models_to_run:
                    models_to_run.append(m)

        out_df = pts.copy()
        out_df["date"] = str(day)
        out_df["has_2d_scene"] = grid_data["has_2d_mask"]
        out_df["recent_fires_7d_50km"] = grid_data["HIST_TAB"][:, 2 * 4 + 3].astype(int)
        out_df["recent_fires_30d_50km"] = grid_data["HIST_TAB"][:, 4 * 4 + 3].astype(int)

        from fwi_calculator import set_scoring_context

        set_scoring_context(day, pts[["lat", "lon"]].values)

        outputs: Dict[str, Dict[str, Any]] = {}
        summary_rows = []
        for m_name in models_to_run:
            try:
                res = gpu_predict(
                    reg=reg,
                    model_name=m_name,
                    tab_1d=grid_data["TAB_1D"],
                    hist_tab=grid_data["HIST_TAB"],
                    clim_feats=grid_data["CLIM_FEATS"],
                    seq=grid_data["SEQ"],
                    static_1d=grid_data["STATIC_1D"],
                    pts_ll=pts[["lat", "lon"]].values,
                    ft_2d=grid_data["FT_2D"],
                    emb32_2d=grid_data["EMB32_2D"],
                    mask=grid_data["has_2d_mask"],
                )
                outputs[m_name] = res
                out_df[f"P_{m_name}"] = np.round(res["p"], 4)
                out_df[f"level_{m_name}"] = res["levels"]
                out_df[f"expected_50km_{m_name}"] = np.round(res["mu"][50], 2)
                out_df[f"upper90_50km_{m_name}"] = res["upper90"][50]

                t = res["thresholds"]
                summary_rows.append(
                    {
                        "Model": m_name,
                        "Description": MODEL_CATALOG.get(m_name, {}).get("label", m_name),
                        "2D Imagery": "Yes" if MODEL_CATALOG.get(m_name, {}).get("requires_2d") else "No (1D/History)",
                        "Mean P(50km)": round(float(res["p"].mean()), 4),
                        "Max P(50km)": round(float(res["p"].max()), 4),
                        "Low (%)": round(100.0 * float((res["levels"] == "Low").mean()), 1),
                        "Medium (%)": round(100.0 * float((res["levels"] == "Medium").mean()), 1),
                        "High (%)": round(100.0 * float((res["levels"] == "High").mean()), 1),
                        "High (count)": int((res["levels"] == "High").sum()),
                        "Mean E[cnt 50km]": round(float(res["mu"][50].mean()), 2),
                        "Threshold Low/Med": round(float(t[0]), 3),
                        "Threshold Med/High": round(float(t[1]), 3),
                    }
                )
            except Exception as e:
                summary_rows.append(
                    {
                        "Model": m_name,
                        "Description": f"Error: {e}",
                        "2D Imagery": "-",
                        "Mean P(50km)": np.nan,
                        "Max P(50km)": np.nan,
                        "Low (%)": np.nan,
                        "Medium (%)": np.nan,
                        "High (%)": np.nan,
                        "High (count)": 0,
                        "Mean E[cnt 50km]": np.nan,
                        "Threshold Low/Med": np.nan,
                        "Threshold Med/High": np.nan,
                    }
                )

        if model_a not in outputs:
            if outputs:
                model_a = next(iter(outputs.keys()))
            else:
                return None, None, None, pd.DataFrame(summary_rows), pd.DataFrame(), None, "No selected models could be evaluated. Check the model files in models/."
        if model_b not in outputs:
            model_b = model_a

        out_df["delta_P_A_minus_B"] = np.round(outputs[model_a]["p"] - outputs[model_b]["p"], 4)
        pred_csv_path = store.save_daily_predictions(day, out_df)

        map_a_path = store.maps_dir / f"map_{model_a}_{day}.png"
        map_b_path = store.maps_dir / f"map_{model_b}_{day}.png"
        cmp_path = store.maps_dir / f"compare_{model_a}_vs_{model_b}_{day}.png"

        draw_single_map(pts, outputs[model_a], day, map_a_path, has_2d_mask=grid_data["has_2d_mask"])
        draw_single_map(pts, outputs[model_b], day, map_b_path, has_2d_mask=grid_data["has_2d_mask"])
        _, cmp_stats = draw_comparison_map(
            pts, outputs[model_a], outputs[model_b], day, cmp_path, has_2d_mask=grid_data["has_2d_mask"]
        )

        summary_df = pd.DataFrame(summary_rows)
        top_df = out_df.sort_values(by=f"P_{model_a}", ascending=False).head(50).reset_index(drop=True)

        info_text = (
            f"Evaluation completed for {day} ({len(pts)} grid locations).\n"
            f"Comparison: {model_a} vs {model_b}: level agreement = {cmp_stats['level_agreement_pct']}%, "
            f"correlation r = {cmp_stats['pearson_r']}, mean absolute difference = {cmp_stats['mean_abs_diff']}.\n"
            f"FIRMS detections past 30 days: {grid_data['n_firms_detections']} detections across {grid_data['n_covered_days']}/30 covered days. "
            f"2D scenes available in cache: {grid_data['n_2d_scenes']}.\n"
            f"Predictions saved to: {pred_csv_path}"
        )

        progress(1.0, desc="Done")
        return str(cmp_path), str(map_a_path), str(map_b_path), summary_df, top_df, str(pred_csv_path), info_text
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, None, None, pd.DataFrame(), pd.DataFrame(), None, f"Error during Iberia grid evaluation: {type(exc).__name__}: {exc}"


def _render_2d_patch_diagnostic(scene_2d: Dict[str, Any], lat: float, lon: float, day_str: str, out_path: Path) -> Path:
    x = np.asarray(scene_2d["X_2d"], dtype=np.float32)
    x32 = build_2d_np(x)[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140)
    ax = axes.ravel()

    rgb = np.stack([x[2], x[1], x[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    if np.percentile(np.abs(rgb), 99) > 2.0:
        rgb = rgb / 10000.0
    rgb_disp = np.clip(rgb / 0.30, 0.0, 1.0) ** 0.85
    ax[0].imshow(rgb_disp)
    ax[0].set_title(f"Sentinel-2 RGB ({scene_2d.get('s2_Source_Date', 'N/A')})")

    im1 = ax[1].imshow(np.nan_to_num(x32[CI["NDVI"]], 0.0), cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    ax[1].set_title("NDVI")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(np.nan_to_num(x32[CI["NBR"]], 0.0), cmap="Spectral", vmin=-0.5, vmax=0.7)
    ax[2].set_title("NBR")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    im3 = ax[3].imshow(np.nan_to_num(x32[CI["NDMI"]], 0.0), cmap="BrBG", vmin=-0.5, vmax=0.6)
    ax[3].set_title("NDMI")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    im4 = ax[4].imshow(np.nan_to_num(x32[CI["SAR_VV"]], 0.0), cmap="gray")
    ax[4].set_title("Sentinel-1 SAR VV (dB)")
    plt.colorbar(im4, ax=ax[4], fraction=0.046)

    im5 = ax[5].imshow(np.nan_to_num(x32[CI["Elevation"]], 0.0), cmap="terrain")
    ax[5].set_title("Copernicus DEM (m)")
    plt.colorbar(im5, ax=ax[5], fraction=0.046)

    im6 = ax[6].imshow(np.nan_to_num(x32[CI["Slope"]], 0.0), cmap="magma")
    ax[6].set_title("Slope (deg)")
    plt.colorbar(im6, ax=ax[6], fraction=0.046)

    im7 = ax[7].imshow(np.nan_to_num(x32[CI["Travel_Time_Roads"]], 0.0), cmap="viridis_r")
    ax[7].set_title("OSM Road Travel Time (h)")
    plt.colorbar(im7, ax=ax[7], fraction=0.046)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.suptitle(f"22-channel 2D satellite patch ({x.shape[1]}x{x.shape[2]} at 100m) - ({lat:.4f}, {lon:.4f}) on {day_str}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _render_single_point_model_chart(preds: Dict[str, Any], lat: float, lon: float, day_str: str, out_path: Path) -> Path:
    valid = [(k, v) for k, v in preds.items() if isinstance(v, dict) and "P_fire_50km" in v]
    valid.sort(key=lambda item: item[1]["P_fire_50km"])

    fig, ax = plt.subplots(figsize=(9.5, max(3.8, 0.52 * max(1, len(valid)))), dpi=140)
    names = [f"{k} {'(2D image)' if v.get('used_2d_imagery') else '(1D)'}" for k, v in valid]
    probs = [v["P_fire_50km"] for _, v in valid]
    bar_colors = [
        "#a50026" if v["risk_level"] == "High" else ("#f46d43" if v["risk_level"] == "Medium" else "#fee08b")
        for _, v in valid
    ]

    if valid:
        bars = ax.barh(names, probs, color=bar_colors, edgecolor="#333333", height=0.65)
        for bar, (_, v) in zip(bars, valid):
            w = bar.get_width()
            ax.text(
                min(w + 0.015, 0.96),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f} ({v['risk_level']}, E[50km]={v['expected_count']['50km']})",
                va="center",
                fontsize=8.5,
            )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("P(fire within 50 km)")
    ax.set_title(f"Model Comparison for ({lat:.4f}, {lon:.4f}) on {day_str}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def run_single_point_analysis(
    lat: float,
    lon: float,
    date_str: str,
    collect_2d: bool,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        day_parsed = _parse_day(date_str)
        progress(0.05, desc="[Step 1/6] Starting single-location pipeline...")
        res = score_single_location(
            lat=float(lat),
            lon=float(lon),
            target_day_str=day_parsed.isoformat(),
            models=["all"],
            collect_2d=bool(collect_2d),
            save_to_mini_dataset=True,
            progress_cb=lambda frac, desc: progress(frac, desc=desc),
        )
        _, store = _get_registry_and_store()
        preds = res["predictions"]

        rows = []
        errors = []
        for m, v in preds.items():
            if "P_fire_50km" in v:
                rows.append(
                    {
                        "Model": m,
                        "Type": "2D Image + 1D" if v.get("used_2d_imagery") else "1D + History/Clim",
                        "P(fire <= 50km)": v["P_fire_50km"],
                        "Risk Level": v["risk_level"],
                        "E[5km]": v["expected_count"]["5km"],
                        "E[10km]": v["expected_count"]["10km"],
                        "E[25km]": v["expected_count"]["25km"],
                        "E[50km]": v["expected_count"]["50km"],
                        "Upper 90% (50km)": v["upper_90pct_count"]["50km"],
                        "Thresholds (Low/Med/High)": f"{v['thresholds_low_med_high'][0]:.2f} / {v['thresholds_low_med_high'][1]:.2f}",
                    }
                )
            elif "error" in v:
                errors.append(f"{m}: {v['error']}")

        if rows:
            df_out = pd.DataFrame(rows).sort_values(by="P(fire <= 50km)", ascending=False).reset_index(drop=True)
        else:
            df_out = pd.DataFrame(columns=["Model", "Type", "P(fire <= 50km)", "Risk Level", "E[5km]", "E[10km]", "E[25km]", "E[50km]", "Upper 90% (50km)", "Thresholds (Low/Med/High)"])

        chart_path = store.maps_dir / f"point_models_{lat:.3f}_{lon:.3f}_{res['date']}.png"
        _render_single_point_model_chart(preds, float(lat), float(lon), res["date"], chart_path)

        patch_img_path = None
        if res.get("scene_2d") is not None:
            patch_img_path = store.maps_dir / f"patch2d_{lat:.3f}_{lon:.3f}_{res['date']}.png"
            _render_2d_patch_diagnostic(res["scene_2d"], float(lat), float(lon), res["date"], patch_img_path)

        status_2d = "collected" if res["has_2d_scene"] else "not collected"
        if res.get("scene_2d_error"):
            status_2d += f" (2D warning: {res['scene_2d_error']})"

        info_lines = [
            f"Location: ({lat:.4f}, {lon:.4f}) on {res['date']}",
            f"2D satellite patch (22 channels): {status_2d} (Sentinel-2 source date: {res.get('s2_source_date')})",
            f"FIRMS detections within 50 km: 7-day count = {res['recent_fires_7d_50km']}, 30-day count = {res['recent_fires_30d_50km']}",
            f"Saved scene path: {res.get('saved_npz_path') or 'none'}",
        ]
        if errors:
            info_lines.append("Model notes: " + "; ".join(errors))

        progress(1.0, desc="Done")
        return str(chart_path), (str(patch_img_path) if patch_img_path else None), df_out, "\n".join(info_lines)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, None, pd.DataFrame(), f"Error during single-location evaluation: {type(exc).__name__}: {exc}"


def run_post_hoc_ground_truth(
    date_str: str,
    model_name: str,
    radius_km: int,
    focal_lat: float,
    focal_lon: float,
    local_radius_km: float,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        day = _parse_day(date_str)
        r_km = int(radius_km)
        reg, store = _get_registry_and_store()
        pts_path = _find_points_csv()
        pts, meta_df = load_points(pts_path)

        cache_key = f"{day}_{len(pts)}"
        if cache_key not in _GRID_CACHE:
            progress(0.05, desc=f"[Step 1/3] Computing pre-fire features strictly up to {day} - 1 day (zero leakage)...")
            grid_data = build_grid_features(
                pts=pts,
                day=day,
                meta_df=meta_df,
                collect_2d_limit=0,
                collect_2d_strategy="top_risk",
                store=store,
                reg=reg,
                progress_cb=lambda frac, desc: progress(0.05 + 0.50 * frac, desc=desc),
            )
            _GRID_CACHE[cache_key] = grid_data
        else:
            progress(0.30, desc=f"[Step 1/3] Loaded pre-fire features for {day} (strictly up to {day} - 1 day)...")
            grid_data = _GRID_CACHE[cache_key]

        progress(0.58, desc="[Step 2/3] Running model predictions before loading ground-truth detections...")
        from fwi_calculator import set_scoring_context

        set_scoring_context(day, pts[["lat", "lon"]].values)

        models_to_eval = [model_name] if model_name else []
        for m in reg.available_models(include_2d=True):
            if m not in models_to_eval:
                models_to_eval.append(m)

        model_preds: Dict[str, Dict[str, Any]] = {}
        model_errors: Dict[str, str] = {}
        for m in models_to_eval:
            try:
                model_preds[m] = gpu_predict(
                    reg=reg,
                    model_name=m,
                    tab_1d=grid_data["TAB_1D"],
                    hist_tab=grid_data["HIST_TAB"],
                    clim_feats=grid_data["CLIM_FEATS"],
                    seq=grid_data["SEQ"],
                    static_1d=grid_data["STATIC_1D"],
                    pts_ll=pts[["lat", "lon"]].values,
                    ft_2d=grid_data["FT_2D"],
                    emb32_2d=grid_data["EMB32_2D"],
                    mask=grid_data["has_2d_mask"],
                )
            except Exception as exc:
                model_errors[m] = f"{type(exc).__name__}: {exc}"
                print(f"  Post-hoc model {m} failed: {model_errors[m]}")

        if not model_preds:
            err_msg = "; ".join(f"{k}: {v}" for k, v in model_errors.items())
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"No models could be evaluated ({err_msg})."

        requested_model = model_name
        if model_name not in model_preds:
            model_name = next(iter(model_preds.keys()))

        progress(0.72, desc=f"[Step 3/3] Post-hoc fetching actual FIRMS detections on {day} and 7-day window...")
        firms_gt = fetch_post_hoc_firms(day, progress_cb=None)
        fire_1d_df = firms_gt["fire_1d_df"]
        fire_7d_df = firms_gt["fire_7d_df"]
        end_7d = firms_gt["end_7d"]

        counts_1d = compute_actual_counts_per_point(pts, fire_1d_df, radii=RADII)
        counts_7d = compute_actual_counts_per_point(pts, fire_7d_df, radii=RADII)

        prim_out = model_preds[model_name]
        gt_map_path = store.maps_dir / f"ground_truth_{model_name}_{day}_R{r_km}km.png"
        draw_ground_truth_comparison_map(
            pts=pts,
            levels=prim_out["levels"],
            probs=prim_out["p"],
            thresholds=prim_out["thresholds"],
            model_name=model_name,
            target_day=day,
            fire_1d_df=fire_1d_df,
            fire_7d_df=fire_7d_df,
            actual_1d_r=counts_1d[r_km],
            actual_7d_r=counts_7d[r_km],
            radius_km=r_km,
            out_path=gt_map_path,
        )

        df_rel_1d = evaluate_tier_reliability_table(
            prim_out["levels"], prim_out["p"], counts_1d[r_km], model_name, f"1d ({day})", r_km
        )
        df_rel_7d = evaluate_tier_reliability_table(
            prim_out["levels"], prim_out["p"], counts_7d[r_km], model_name, f"7d ({day}..{end_7d})", r_km
        )
        df_reliability_gt = pd.concat([df_rel_1d, df_rel_7d], ignore_index=True)

        model_rows = []
        for m, out in model_preds.items():
            model_rows.append(
                evaluate_levels_against_actual(
                    levels=out["levels"],
                    probs=out["p"],
                    actual_counts=counts_1d[r_km],
                    model_name=m,
                    window_label=f"1d ({day})",
                    radius_km=r_km,
                )
            )
            model_rows.append(
                evaluate_levels_against_actual(
                    levels=out["levels"],
                    probs=out["p"],
                    actual_counts=counts_7d[r_km],
                    model_name=m,
                    window_label=f"7d ({day}..{end_7d})",
                    radius_km=r_km,
                )
            )
        df_models_gt = evaluate_vs_fwi_benchmark(pd.DataFrame(model_rows))

        radii_rows = []
        for R in RADII:
            radii_rows.append(
                evaluate_levels_against_actual(
                    levels=prim_out["levels"],
                    probs=prim_out["p"],
                    actual_counts=counts_1d[int(R)],
                    model_name=model_name,
                    window_label=f"1d ({day})",
                    radius_km=int(R),
                )
            )
            radii_rows.append(
                evaluate_levels_against_actual(
                    levels=prim_out["levels"],
                    probs=prim_out["p"],
                    actual_counts=counts_7d[int(R)],
                    model_name=model_name,
                    window_label=f"7d ({day}..{end_7d})",
                    radius_km=int(R),
                )
            )
        df_radii_gt = pd.DataFrame(radii_rows)

        local_rows = []
        for m, out in model_preds.items():
            local_rows.append(
                evaluate_local_radius_around_point(
                    pts=pts,
                    center_lat=float(focal_lat),
                    center_lon=float(focal_lon),
                    search_radius_km=float(local_radius_km),
                    levels=out["levels"],
                    probs=out["p"],
                    actual_1d=counts_1d[r_km],
                    actual_7d=counts_7d[r_km],
                    model_name=m,
                )
            )
        df_local_gt = pd.DataFrame(local_rows)

        s1 = evaluate_levels_against_actual(
            prim_out["levels"], prim_out["p"], counts_1d[r_km], model_name, f"1d ({day})", r_km
        )
        s7 = evaluate_levels_against_actual(
            prim_out["levels"], prim_out["p"], counts_7d[r_km], model_name, f"7d ({day}..{end_7d})", r_km
        )

        fwi_cmp_lines: List[str] = []
        if "fwi" in model_preds:
            fwi_out = model_preds["fwi"]
            f1 = evaluate_levels_against_actual(
                fwi_out["levels"], fwi_out["p"], counts_1d[r_km], "fwi", f"1d ({day})", r_km
            )
            f7 = evaluate_levels_against_actual(
                fwi_out["levels"], fwi_out["p"], counts_7d[r_km], "fwi", f"7d ({day}..{end_7d})", r_km
            )
            d_auc1 = (
                round(float(s1["ROC-AUC"]) - float(f1["ROC-AUC"]), 4)
                if np.isfinite(s1["ROC-AUC"]) and np.isfinite(f1["ROC-AUC"])
                else np.nan
            )
            d_auc7 = (
                round(float(s7["ROC-AUC"]) - float(f7["ROC-AUC"]), 4)
                if np.isfinite(s7["ROC-AUC"]) and np.isfinite(f7["ROC-AUC"])
                else np.nan
            )
            fwi_cmp_lines = [
                f"",
                f"[Head-to-Head vs Canadian FWI (EFFIS Physical Baseline) at R = {r_km} km]:",
                f"  - 1d ({day}): {model_name} ROC-AUC = {s1['ROC-AUC']} vs FWI ROC-AUC = {f1['ROC-AUC']} (Delta = {d_auc1:+.4f}) | {model_name} High/Low = {s1['High/Low Rate Ratio (x)']}x vs FWI High/Low = {f1['High/Low Rate Ratio (x)']}x",
                f"  - 7d ({day}..{end_7d}): {model_name} ROC-AUC = {s7['ROC-AUC']} vs FWI ROC-AUC = {f7['ROC-AUC']} (Delta = {d_auc7:+.4f}) | {model_name} High/Low = {s7['High/Low Rate Ratio (x)']}x vs FWI High/Low = {f7['High/Low Rate Ratio (x)']}x",
            ]

        diag = firms_gt.get("status_info", {})
        firms_diag_lines: List[str] = [f""]
        if requested_model and requested_model != model_name and requested_model in model_errors:
            firms_diag_lines.append(f"⚠️ Model `{requested_model}` failed ({model_errors[requested_model]}); displaying `{model_name}` instead.")
        if diag.get("last_error"):
            firms_diag_lines.append(f"⚠️ NASA FIRMS API Warning: {diag['last_error']}")
        if len(fire_1d_df) == 0 and len(fire_7d_df) == 0:
            if diag.get("last_error") or not diag.get("keys_configured"):
                firms_diag_lines.append("⚠️ Notice: 0 ground-truth detections loaded because FIRMS API keys were exhausted, invalid, or missing. Check the FIRMS API key pool.")
            else:
                firms_diag_lines.append("ℹ️ Notice: NASA FIRMS reported 0 active fire detections across Iberia during this window.")
        else:
            k_pref = diag.get("active_key_prefix", "none")
            s_used = diag.get("last_src", "archive/cache")
            n_keys = diag.get("keys_configured", 0)
            firms_diag_lines.append(f"ℹ️ FIRMS Status: Active key {k_pref} | Source {s_used} | Key pool size: {n_keys}")

        status_lines = [
            f"Stochastic & Probabilistic Verification for {day} (strictly isolated from pre-fire features up to {day} - 1 day):",
            f"Raw VIIRS fire detections in Iberia: {len(fire_1d_df)} on {day} (1d) | {len(fire_7d_df)} across {day}..{end_7d} (7d).",
            f"",
            f"[{model_name} | Radius R = {r_km} km | Target Day ({day})]:",
            f"  - Empirical Fire Rate by Tier: Low = {s1['P(fire | Low) (%)']}% -> Medium = {s1['P(fire | Med) (%)']}% -> High = {s1['P(fire | High) (%)']}% (Base Rate = {s1['Base Fire Rate (%)']}%)",
            f"  - Risk Stratification: High/Low Ratio = {s1['High/Low Rate Ratio (x)']}x | Monotonic = {s1['Monotonic (Low<=Med<=High)']} | Lift(High) = {s1['Lift High (x)']}x | Lift(Med+High) = {s1['Lift Med+High (x)']}x",
            f"  - Threshold-Free Metrics: ROC-AUC = {s1['ROC-AUC']} | PR-AUC = {s1['PR-AUC']} | Brier Score = {s1['Brier Score']}",
            f"  - Low-Tier Safety: {s1['Low Safe Rate (%)']}% fire-free | Recall(Med+High) = {s1['Recall Med+High (%)']}% (Missed in Low = {s1['Missed in Low (%)']}%)",
            f"",
            f"[{model_name} | Radius R = {r_km} km | 7-Day Window ({day} to {end_7d})]:",
            f"  - Empirical Fire Rate by Tier: Low = {s7['P(fire | Low) (%)']}% -> Medium = {s7['P(fire | Med) (%)']}% -> High = {s7['P(fire | High) (%)']}% (Base Rate = {s7['Base Fire Rate (%)']}%)",
            f"  - Risk Stratification: High/Low Ratio = {s7['High/Low Rate Ratio (x)']}x | Monotonic = {s7['Monotonic (Low<=Med<=High)']} | Lift(High) = {s7['Lift High (x)']}x | Lift(Med+High) = {s7['Lift Med+High (x)']}x",
            f"  - Threshold-Free Metrics: ROC-AUC = {s7['ROC-AUC']} | PR-AUC = {s7['PR-AUC']} | Brier Score = {s7['Brier Score']}",
            f"  - Low-Tier Safety: {s7['Low Safe Rate (%)']}% fire-free | Recall(Med+High) = {s7['Recall Med+High (%)']}% (Missed in Low = {s7['Missed in Low (%)']}%)",
        ] + fwi_cmp_lines + firms_diag_lines

        progress(1.0, desc="Done")
        return str(gt_map_path), df_reliability_gt, df_models_gt, df_radii_gt, df_local_gt, "\n".join(status_lines)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Error in post-hoc evaluation: {type(exc).__name__}: {exc}"


def run_batch_post_hoc_evaluation(
    dates_csv_str: str,
    radius_km: int,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        raw_parts = [p.strip()[:10] for p in str(dates_csv_str or "").replace("\n", ",").split(",") if p.strip()]
        days: List[Any] = []
        for p in raw_parts:
            try:
                d = datetime.strptime(p, "%Y-%m-%d").date()
                if d not in days:
                    days.append(d)
            except Exception:
                pass
        if not days:
            return pd.DataFrame(), pd.DataFrame(), "No valid YYYY-MM-DD dates provided."

        r_km = int(radius_km)
        reg, store = _get_registry_and_store()
        pts_path = _find_points_csv()
        pts, meta_df = load_points(pts_path)
        avail_models = reg.available_models(include_2d=False)

        from fwi_calculator import set_scoring_context

        per_date_rows = []
        for idx, day in enumerate(days):
            frac_base = idx / len(days)
            span = 1.0 / len(days)
            progress(frac_base + 0.05 * span, desc=f"[Batch {idx + 1}/{len(days)}] Features for {day} (strictly up to {day} - 1d)...")

            cache_key = f"{day}_{len(pts)}"
            if cache_key not in _GRID_CACHE:
                grid_data = build_grid_features(
                    pts=pts,
                    day=day,
                    meta_df=meta_df,
                    collect_2d_limit=0,
                    collect_2d_strategy="top_risk",
                    store=store,
                    reg=reg,
                    progress_cb=None,
                )
                _GRID_CACHE[cache_key] = grid_data
            else:
                grid_data = _GRID_CACHE[cache_key]

            set_scoring_context(day, pts[["lat", "lon"]].values)

            progress(frac_base + 0.55 * span, desc=f"[Batch {idx + 1}/{len(days)}] Fetching ground-truth FIRMS for {day} (1d & 7d)...")
            firms_gt = fetch_post_hoc_firms(day, progress_cb=None)
            c1 = compute_actual_counts_per_point(pts, firms_gt["fire_1d_df"], radii=(r_km,))[r_km]
            c7 = compute_actual_counts_per_point(pts, firms_gt["fire_7d_df"], radii=(r_km,))[r_km]

            day_rows_1d = []
            day_rows_7d = []
            for m in avail_models:
                try:
                    out = gpu_predict(
                        reg=reg,
                        model_name=m,
                        tab_1d=grid_data["TAB_1D"],
                        hist_tab=grid_data["HIST_TAB"],
                        clim_feats=grid_data["CLIM_FEATS"],
                        seq=grid_data["SEQ"],
                        static_1d=grid_data["STATIC_1D"],
                        pts_ll=pts[["lat", "lon"]].values,
                        ft_2d=grid_data["FT_2D"],
                        emb32_2d=grid_data["EMB32_2D"],
                        mask=grid_data["has_2d_mask"],
                    )
                    r1 = evaluate_levels_against_actual(out["levels"], out["p"], c1, m, "1d", r_km)
                    r1["Date"] = str(day)
                    day_rows_1d.append(r1)

                    r7 = evaluate_levels_against_actual(out["levels"], out["p"], c7, m, "7d", r_km)
                    r7["Date"] = str(day)
                    day_rows_7d.append(r7)
                except Exception as exc:
                    print(f"  Batch post-hoc model {m} on {day} failed: {exc}")

            if day_rows_1d:
                df_d1 = evaluate_vs_fwi_benchmark(pd.DataFrame(day_rows_1d))
                per_date_rows.extend(df_d1.to_dict("records"))
            if day_rows_7d:
                df_d7 = evaluate_vs_fwi_benchmark(pd.DataFrame(day_rows_7d))
                per_date_rows.extend(df_d7.to_dict("records"))

        if not per_date_rows:
            return pd.DataFrame(), pd.DataFrame(), "No models/dates could be evaluated."

        df_per_date = pd.DataFrame(per_date_rows)
        num_cols = [
            "Base Fire Rate (%)",
            "P(fire | Low) (%)",
            "P(fire | Med) (%)",
            "P(fire | High) (%)",
            "High/Low Rate Ratio (x)",
            "Lift High (x)",
            "Lift Med+High (x)",
            "Recall Med+High (%)",
            "Recall High (%)",
            "Missed in Low (%)",
            "Low Safe Rate (%)",
            "ROC-AUC",
            "Delta ROC-AUC vs FWI",
            "Skill Gain vs FWI (%)",
            "PR-AUC",
            "Delta PR-AUC vs FWI",
            "Brier Score",
        ]
        agg_rows = []
        for (m, win), grp in df_per_date.groupby(["Model", "Window"], sort=False):
            row: Dict[str, Any] = {
                "Model": m,
                "Window": win,
                "Dates Evaluated": len(grp),
                "Radius (km)": r_km,
            }
            for col in num_cols:
                if col in grp.columns:
                    val = grp[col].dropna()
                    row[f"Mean {col}"] = round(float(val.mean()), 4 if "AUC" in col or "Brier" in col else 2) if len(val) else np.nan
            mono_rate = 100.0 * float((grp["Monotonic (Low<=Med<=High)"] == "Yes").mean())
            row["Monotonic Dates (%)"] = round(mono_rate, 1)
            agg_rows.append(row)

        df_agg = pd.DataFrame(agg_rows).sort_values(by=["Window", "Mean ROC-AUC"], ascending=[True, False]).reset_index(drop=True)
        cols_order = ["Date"] + [c for c in df_per_date.columns if c != "Date"]
        df_per_date = df_per_date[cols_order]

        progress(1.0, desc="Batch evaluation complete")
        summary_msg = (
            f"Completed multi-date stochastic evaluation across {len(days)} dates ({', '.join(str(d) for d in days)}) at R = {r_km} km.\n"
            f"Evaluated {len(avail_models)} models ({len(df_per_date)} model-date-window runs)."
        )
        return df_agg, df_per_date, summary_msg
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), f"Error in batch evaluation: {type(exc).__name__}: {exc}"


def refresh_mini_dataset_view():
    _, store = _get_registry_and_store()
    st = store.summary_stats()
    meta_df = pd.read_csv(store.metadata_path) if store.metadata_path.exists() else pd.DataFrame()
    info_text = (
        f"Mini-dataset status ({st['root_dir']}):\n"
        f"Saved 2D/1D scenes (.npz): {st['num_npz_scenes']}\n"
        f"Saved daily feature tables: {st['num_feature_days']}\n"
        f"Saved prediction tables: {st['num_prediction_days']}\n"
        f"Saved maps: {st['num_saved_maps']}\n"
        f"Total disk usage: {st['total_size_mb']} MB"
    )
    return info_text, meta_df


def create_mini_dataset_zip():
    _, store = _get_registry_and_store()
    zip_path = store.export_zip()
    return str(zip_path)


def get_system_status_markdown() -> Tuple[str, pd.DataFrame]:
    reg, _ = _get_registry_and_store()
    rep = reg.status_report()
    rows = []
    for k, st in rep.items():
        rows.append(
            {
                "Model Key": k,
                "Name": st["label"],
                "Status": "ready" if st["available"] else "weights missing",
                "Requires 2D (.npz)": "Yes" if st["requires_2d"] else "No",
                "Missing Files": ", ".join(st["missing_files"]) if st["missing_files"] else "none",
            }
        )
    df_models = pd.DataFrame(rows)

    k_pool = get_firms_keys()
    k_label = f"FIRMS_MAP_KEY ({len(k_pool)} active keys configured in pool)" if k_pool else "FIRMS_MAP_KEY secret"
    data_checks = [
        ("metadata.csv", Path(_find_points_csv()).exists()),
        ("firms_covered_days.csv", Path(CONFIG["HISTORICAL_COVERED_CSV"]).exists() or Path("data/firms_covered_days.csv").exists()),
        ("firms_points_2017_2025.csv.gz", Path(CONFIG["HISTORICAL_FIRMS_CSV"]).exists() or Path("data/firms_points_2017_2025.csv.gz").exists()),
        ("oof_predictions.csv", reg.oof_path is not None and reg.oof_path.exists()),
        ("population_cache.csv", Path(CONFIG["POPULATION_CACHE_CSV"]).exists()),
        (k_label, bool(k_pool)),
    ]
    lines = ["System file and configuration checks:"]
    for label, ok in data_checks:
        lines.append(f"- {label}: {'present' if ok else 'missing'}")
    return "\n".join(lines), df_models


def build_app() -> gr.Blocks:
    choices = _model_choices()
    default_a = "tab_clim_full"
    default_b = "nn_ts" if any(k == "nn_ts" for _, k in choices) else "persist"

    with gr.Blocks(title="PreemptFire Iberia Wildfire Risk") as demo:
        gr.Markdown(
            "# PreemptFire: Wildfire Risk Assessment and Model Comparison (Iberia)\n\n"
            "**Benchmark Dataset (Kaggle):** [Wildfire Iberia 100m](https://www.kaggle.com/datasets/hollyfrogvi/wildfire-iberia-100m) | "
            "**Source Code:** [GitHub Repository](https://github.com/tallchildvi/PreemptFire)\n\n"
            "Interactive interface for running multi-model wildfire predictions (1D meteorological, GRU sequence models, and 2D satellite models) "
            "with dataset persistence and isolated post-hoc ground-truth verification."
        )

        with gr.Tabs():
            with gr.TabItem("1. Iberia Grid Map and Comparison"):
                with gr.Row():
                    date_in = gr.Textbox(value="2025-08-12", label="Target Date (YYYY-MM-DD)", scale=1)
                    model_a_in = gr.Dropdown(choices=choices, value=default_a, label="Model A (Primary)", scale=2)
                    model_b_in = gr.Dropdown(choices=choices, value=default_b, label="Model B (Comparison)", scale=2)

                with gr.Accordion("2D Scene Collection and Caching Options", open=False):
                    with gr.Row():
                        also_all_in = gr.Checkbox(value=True, label="Also compute metrics for all available models in summary table")
                        collect_2d_limit_in = gr.Slider(
                            minimum=0,
                            maximum=30,
                            step=1,
                            value=0,
                            label="Collect N live 2D scenes for this date (0 = use cached only)",
                        )
                        collect_2d_strat_in = gr.Radio(
                            choices=["top_risk", "random"],
                            value="top_risk",
                            label="Location selection strategy for 2D collection",
                        )
                        force_refresh_in = gr.Checkbox(value=False, label="Force refresh weather and FIRMS cache")

                run_map_btn = gr.Button("Compute Maps and Compare Models", variant="primary")
                map_status_out = gr.Textbox(label="Run Status", interactive=False)

                gr.Markdown("### Comparison: Model A vs Model B and Difference Map (Delta P = P_A - P_B)")
                cmp_img_out = gr.Image(label="Comparison Map", type="filepath")

                with gr.Row():
                    map_a_out = gr.Image(label="Model A Map (Risk Level + Expected Count 50km)", type="filepath")
                    map_b_out = gr.Image(label="Model B Map (Risk Level + Expected Count 50km)", type="filepath")

                gr.Markdown("### Summary Table (All Evaluated Models)")
                summary_table_out = gr.Dataframe(label="Model Comparison on Iberia Grid")

                gr.Markdown("### Top 50 Locations by Risk")
                top_table_out = gr.Dataframe(label="Top 50 Locations")
                csv_download_out = gr.File(label="Download Predictions CSV")

                run_map_btn.click(
                    fn=run_iberia_map_and_compare,
                    inputs=[
                        date_in,
                        model_a_in,
                        model_b_in,
                        also_all_in,
                        collect_2d_limit_in,
                        collect_2d_strat_in,
                        force_refresh_in,
                    ],
                    outputs=[
                        cmp_img_out,
                        map_a_out,
                        map_b_out,
                        summary_table_out,
                        top_table_out,
                        csv_download_out,
                        map_status_out,
                    ],
                )

            with gr.TabItem("2. Single Location and 2D Satellite Patch"):
                gr.Markdown(
                    "Collect a 22-channel 256x256 satellite patch (Sentinel-2, Sentinel-1, DEM, OSM road travel time, ERA5 soil moisture) "
                    "for any coordinate, store it in the mini-dataset, and evaluate 2D models alongside 1D models."
                )
                with gr.Row():
                    pt_lat_in = gr.Number(value=40.15, label="Latitude (e.g. 36.0 to 43.8)")
                    pt_lon_in = gr.Number(value=-4.65, label="Longitude (e.g. -9.3 to 3.2)")
                    pt_date_in = gr.Textbox(value="2025-08-12", label="Date (YYYY-MM-DD)")
                    pt_2d_in = gr.Checkbox(
                        value=True,
                        label="Collect 22-channel 2D patch (enables tab_ssl_full and tab_full)",
                    )

                run_pt_btn = gr.Button("Evaluate Single Location", variant="primary")
                pt_status_out = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    pt_chart_out = gr.Image(label="Predicted P(fire <= 50km) Across Models", type="filepath")
                    pt_patch_out = gr.Image(label="2D Satellite Patch Channels", type="filepath")

                pt_table_out = gr.Dataframe(label="Prediction Table (5km, 10km, 25km, 50km)")

                run_pt_btn.click(
                    fn=run_single_point_analysis,
                    inputs=[pt_lat_in, pt_lon_in, pt_date_in, pt_2d_in],
                    outputs=[pt_chart_out, pt_patch_out, pt_table_out, pt_status_out],
                )

            with gr.TabItem("3. Ground Truth Fire Detections (Post-Hoc Verification)"):
                gr.Markdown(
                    "Independent post-hoc evaluation module (`post_hoc_evaluation.py`). "
                    "Predictions are computed strictly using data up to `Target Date - 1 day` (zero data leakage). "
                    "Because wildfire ignition is a stochastic process, verification measures **empirical fire frequency by risk tier** "
                    "(`P(fire | Low) < P(fire | Medium) < P(fire | High)`), **Lift vs Baseline**, **ROC-AUC**, **PR-AUC**, **Brier Score**, "
                    "and **Low-tier safety rate** across the Target Day (1d) and 7-Day Forward Window (7d)."
                )
                with gr.Row():
                    gt_date_in = gr.Textbox(value="2025-08-12", label="Target Date (YYYY-MM-DD)", scale=1)
                    gt_model_in = gr.Dropdown(choices=choices, value=default_a, label="Primary Model for Map & Tier Reliability", scale=2)
                    gt_radius_in = gr.Radio(choices=[5, 10, 25, 50], value=50, label="Fire Verification Radius R (km)", scale=1)

                with gr.Row():
                    gt_lat_in = gr.Number(value=40.15, label="Local Check Latitude")
                    gt_lon_in = gr.Number(value=-4.65, label="Local Check Longitude")
                    gt_loc_rad_in = gr.Slider(minimum=5, maximum=150, step=5, value=50, label="Local Coordinate Check Radius (km)")

                run_gt_btn = gr.Button("Fetch Actual Detections (1d & 7d) and Compute Reliability & Lift", variant="primary")
                gt_status_out = gr.Textbox(label="Probabilistic Calibration & Risk Stratification Summary", lines=13, interactive=False)

                gr.Markdown("### Predicted Risk Field vs Actual Fire Detections (Target Day & 7-Day Window)")
                gt_map_out = gr.Image(label="Ground-Truth Comparison Map (Predicted Field vs 1d & 7d Empirical Rates)", type="filepath")

                gr.Markdown("### Tier Reliability & Lift Table for Selected Primary Model (Low / Medium / High / Med+High)")
                gt_rel_table_out = gr.Dataframe(label="Empirical Fire Frequency P(fire | Tier), Lift vs Baseline, and Share of Total Fires")

                gr.Markdown("### All Models: Calibration, Monotonicity, Lift, ROC-AUC, PR-AUC, and Brier Score")
                gt_models_table_out = gr.Dataframe(label="All Models Probabilistic Verification (1d and 7d)")

                gr.Markdown("### Multi-Radius Breakdown (5 km, 10 km, 25 km, 50 km) for Selected Primary Model")
                gt_radii_table_out = gr.Dataframe(label="Primary Model Breakdown Across All Radii")

                gr.Markdown("### Local Radius Risk Distribution & Actual Fire Rate Around Selected Coordinate")
                gt_local_table_out = gr.Dataframe(label="Local Coordinate Radius Check Across Models")

                with gr.Accordion("Multi-Date Batch Evaluation (Average Reliability, Lift, ROC-AUC, PR-AUC, Brier Across Dates)", open=False):
                    gr.Markdown(
                        "Evaluate all 1D/sequence models across multiple dates (e.g. different seasons or fire episodes) "
                        "and compute mean empirical fire rates per tier, mean Lift, ROC-AUC, PR-AUC, and Brier Score."
                    )
                    with gr.Row():
                        batch_dates_in = gr.Textbox(
                            value="2025-07-15, 2025-08-05, 2025-08-12, 2025-08-20",
                            label="Comma-separated Target Dates (YYYY-MM-DD)",
                            scale=3,
                        )
                        batch_rad_in = gr.Radio(choices=[5, 10, 25, 50], value=50, label="Batch Verification Radius R (km)", scale=1)
                    run_batch_gt_btn = gr.Button("Run Multi-Date Batch Evaluation", variant="secondary")
                    batch_gt_status_out = gr.Textbox(label="Batch Evaluation Status", interactive=False)
                    batch_agg_table_out = gr.Dataframe(label="Multi-Date Mean Metrics per Model (Sorted by Mean ROC-AUC)")
                    batch_daily_table_out = gr.Dataframe(label="Per-Date Detailed Breakdown")

                run_gt_btn.click(
                    fn=run_post_hoc_ground_truth,
                    inputs=[gt_date_in, gt_model_in, gt_radius_in, gt_lat_in, gt_lon_in, gt_loc_rad_in],
                    outputs=[gt_map_out, gt_rel_table_out, gt_models_table_out, gt_radii_table_out, gt_local_table_out, gt_status_out],
                )
                run_batch_gt_btn.click(
                    fn=run_batch_post_hoc_evaluation,
                    inputs=[batch_dates_in, batch_rad_in],
                    outputs=[batch_agg_table_out, batch_daily_table_out, batch_gt_status_out],
                )

            with gr.TabItem("4. Mini-Dataset and Export"):
                gr.Markdown(
                    "Collected 2D scenes (.npz), feature tables, predictions, and maps are stored in the mini_dataset/ folder."
                )
                with gr.Row():
                    refresh_ds_btn = gr.Button("Refresh Dataset Status")
                    export_zip_btn = gr.Button("Download Full Mini-Dataset (.zip)", variant="primary")

                ds_status_out = gr.Textbox(label="Mini-Dataset Status", interactive=False)
                zip_file_out = gr.File(label="Mini-Dataset Archive (.zip)")
                ds_meta_table = gr.Dataframe(label="Collected Scenes (metadata.csv)")

                refresh_ds_btn.click(fn=refresh_mini_dataset_view, inputs=[], outputs=[ds_status_out, ds_meta_table])
                export_zip_btn.click(fn=create_mini_dataset_zip, inputs=[], outputs=[zip_file_out])

            with gr.TabItem("5. Model and File Status"):
                check_btn = gr.Button("Check Files")
                sys_md, sys_df = get_system_status_markdown()
                sys_md_out = gr.Textbox(value=sys_md, label="Environment Checks", interactive=False)
                sys_df_out = gr.Dataframe(value=sys_df, label="Model Registry Status")
                check_btn.click(fn=get_system_status_markdown, inputs=[], outputs=[sys_md_out, sys_df_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()
