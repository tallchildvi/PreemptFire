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

from config_and_features import CONFIG, RADII
from live import score_single_location
from make_map import (
    build_grid_features,
    draw_comparison_map,
    draw_single_map,
    load_points,
)
from mini_dataset_store import MiniDatasetStore
from model_registry import COLORS, LEVELS, MODEL_CATALOG, ModelRegistry

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
    day = datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").date()
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

    outputs: Dict[str, Dict[str, Any]] = {}
    summary_rows = []
    for m_name in models_to_run:
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
            raise gr.Error("No selected models could be evaluated. Check the model files in models/.")
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


def _render_2d_patch_diagnostic(scene_2d: Dict[str, Any], lat: float, lon: float, day_str: str, out_path: Path) -> Path:
    x = np.asarray(scene_2d["X_2d"], dtype=np.float32)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140)
    ax = axes.ravel()

    rgb = np.stack([x[2], x[1], x[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb_disp = np.clip(rgb / 0.30, 0.0, 1.0) ** 0.85
    ax[0].imshow(rgb_disp)
    ax[0].set_title(f"Sentinel-2 RGB ({scene_2d.get('s2_Source_Date', 'N/A')})")

    im1 = ax[1].imshow(np.nan_to_num(x[10], 0.0), cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    ax[1].set_title("NDVI")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(np.nan_to_num(x[11], 0.0), cmap="Spectral", vmin=-0.5, vmax=0.7)
    ax[2].set_title("NBR")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    im3 = ax[3].imshow(np.nan_to_num(x[12], 0.0), cmap="BrBG", vmin=-0.5, vmax=0.6)
    ax[3].set_title("NDWI")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    im4 = ax[4].imshow(np.nan_to_num(x[13], 0.0), cmap="gray")
    ax[4].set_title("Sentinel-1 SAR VV")
    plt.colorbar(im4, ax=ax[4], fraction=0.046)

    im5 = ax[5].imshow(np.nan_to_num(x[16], 0.0), cmap="terrain")
    ax[5].set_title("Copernicus DEM (m)")
    plt.colorbar(im5, ax=ax[5], fraction=0.046)

    im6 = ax[6].imshow(np.nan_to_num(x[17], 0.0), cmap="magma")
    ax[6].set_title("Slope (deg)")
    plt.colorbar(im6, ax=ax[6], fraction=0.046)

    im7 = ax[7].imshow(np.nan_to_num(x[20], 0.0), cmap="viridis_r")
    ax[7].set_title("OSM Road Travel Time (min)")
    plt.colorbar(im7, ax=ax[7], fraction=0.046)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.suptitle(f"22-channel 2D satellite patch (256x256 at 10m) - ({lat:.4f}, {lon:.4f}) on {day_str}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _render_single_point_model_chart(preds: Dict[str, Any], lat: float, lon: float, day_str: str, out_path: Path) -> Path:
    valid = [(k, v) for k, v in preds.items() if isinstance(v, dict) and "P_fire_50km" in v]
    valid.sort(key=lambda item: item[1]["P_fire_50km"])

    fig, ax = plt.subplots(figsize=(9.5, max(3.8, 0.52 * len(valid))), dpi=140)
    names = [f"{k} {'(2D image)' if v.get('used_2d_imagery') else '(1D)'}" for k, v in valid]
    probs = [v["P_fire_50km"] for _, v in valid]
    bar_colors = [
        "#a50026" if v["risk_level"] == "High" else ("#f46d43" if v["risk_level"] == "Medium" else "#fee08b")
        for _, v in valid
    ]

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
    progress(0.15, desc="Fetching data...")
    res = score_single_location(
        lat=float(lat),
        lon=float(lon),
        target_day_str=str(date_str).strip()[:10],
        models=["all"],
        collect_2d=bool(collect_2d),
        save_to_mini_dataset=True,
    )
    _, store = _get_registry_and_store()
    preds = res["predictions"]

    rows = []
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
    df_out = pd.DataFrame(rows).sort_values(by="P(fire <= 50km)", ascending=False).reset_index(drop=True)

    chart_path = store.maps_dir / f"point_models_{lat:.3f}_{lon:.3f}_{res['date']}.png"
    _render_single_point_model_chart(preds, float(lat), float(lon), res["date"], chart_path)

    patch_img_path = None
    if res.get("scene_2d") is not None:
        patch_img_path = store.maps_dir / f"patch2d_{lat:.3f}_{lon:.3f}_{res['date']}.png"
        _render_2d_patch_diagnostic(res["scene_2d"], float(lat), float(lon), res["date"], patch_img_path)

    info_text = (
        f"Location: ({lat:.4f}, {lon:.4f}) on {res['date']}\n"
        f"2D satellite patch (22 channels): {'collected' if res['has_2d_scene'] else 'not collected'} "
        f"(Sentinel-2 source date: {res.get('s2_source_date')})\n"
        f"FIRMS detections within 50 km: 7-day count = {res['recent_fires_7d_50km']}, 30-day count = {res['recent_fires_30d_50km']}\n"
        f"Saved scene path: {res.get('saved_npz_path') or 'none'}"
    )
    progress(1.0, desc="Done")
    return str(chart_path), (str(patch_img_path) if patch_img_path else None), df_out, info_text


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

    data_checks = [
        ("metadata.csv", Path(_find_points_csv()).exists()),
        ("firms_covered_days.csv", Path(CONFIG["HISTORICAL_COVERED_CSV"]).exists() or Path("data/firms_covered_days.csv").exists()),
        ("firms_points_2017_2025.csv.gz", Path(CONFIG["HISTORICAL_FIRMS_CSV"]).exists() or Path("data/firms_points_2017_2025.csv.gz").exists()),
        ("oof_predictions.csv", reg.oof_path is not None and reg.oof_path.exists()),
        ("population_cache.csv", Path(CONFIG["POPULATION_CACHE_CSV"]).exists()),
        ("FIRMS_MAP_KEY secret", bool(CONFIG.get("FIRMS_MAP_KEY"))),
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
            "# PreemptFire: Wildfire Risk Assessment and Model Comparison (Iberia)\n"
            "Interactive interface for running multi-model wildfire predictions (1D, GRU sequence models, and 2D satellite models) "
            "with dataset persistence."
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

            with gr.TabItem("3. Mini-Dataset and Export"):
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

            with gr.TabItem("4. Model and File Status"):
                check_btn = gr.Button("Check Files")
                sys_md, sys_df = get_system_status_markdown()
                sys_md_out = gr.Textbox(value=sys_md, label="Environment Checks", interactive=False)
                sys_df_out = gr.Dataframe(value=sys_df, label="Model Registry Status")
                check_btn.click(fn=get_system_status_markdown, inputs=[], outputs=[sys_md_out, sys_df_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()
