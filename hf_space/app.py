"""Gradio Web Application for PreemptFire on Hugging Face Spaces.

Features:
1. 🗺️ Карта пожежної небезпеки Іберії та Візуальне Порівняння Моделей:
   - Select any date and compare any pair of models side-by-side (Model A vs Model B + ΔP map).
   - Supports both fast 1D/History/Climatology/GRU models across all 3,146 locations AND
     live/cached 2D satellite image models (`tab_ssl_full`, `tab_full`).
   - Caches daily features in memory & `mini_dataset/` so switching or comparing models on the same date is instant!
2. 🛰️ Live-Аналіз Точки + 22-канальний 2D Супутниковий Патч (`Sentinel-2 + S1 + DEM + OSM + ERA5`):
   - Collects or loads a 256×256×22 `.npz` scene for any coordinate and date.
   - Visualizes 8 satellite/terrain channels (True Color RGB, NDVI, NBR, NDWI, Sentinel-1 VV, Elevation, Slope, Road Travel Time).
   - Compares all 1D and 2D models side-by-side (probabilities + expected fire counts at 5/10/25/50 km).
3. 📦 Міні-Датасет та Експорт:
   - Automatically stores every collected `.npz` scene, `metadata.csv`, daily tabular features, and model predictions.
   - One-click `.zip` export for downloading the collected mini-dataset and results.
"""
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

# In-memory cache of computed daily grid features so comparing different models on the same date is instant
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
        badge = "✅" if avail else "⚠️ (немає ваг у models/)"
        choices.append((f"{badge} [{k}] {info['label']}", k))
    return choices


# ==============================================================================
# Tab 1: Iberia Risk Map & Multi-Model Visual Comparison
# ==============================================================================

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

    cache_key = f"{day}_{ len(pts)}"
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
        progress(0.35, desc=f"Використання кешованих ознак за {day} (миттєве порівняння моделей)...")
        grid_data = _GRID_CACHE[cache_key]

    progress(0.82, desc="Запуск моделей та побудова карт...")
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
                    "Модель": m_name,
                    "Опис": MODEL_CATALOG.get(m_name, {}).get("label", m_name),
                    "2D Знімки": "Так" if MODEL_CATALOG.get(m_name, {}).get("requires_2d") else "Ні (1D+Історія)",
                    "Mean P(50km)": round(float(res["p"].mean()), 4),
                    "Max P(50km)": round(float(res["p"].max()), 4),
                    "Low (%)": round(100.0 * float((res["levels"] == "Low").mean()), 1),
                    "Medium (%)": round(100.0 * float((res["levels"] == "Medium").mean()), 1),
                    "High (%)": round(100.0 * float((res["levels"] == "High").mean()), 1),
                    "High (к-сть сцен)": int((res["levels"] == "High").sum()),
                    "Mean E[cnt 50km]": round(float(res["mu"][50].mean()), 2),
                    "Поріг Low/Med": round(float(t[0]), 3),
                    "Поріг Med/High": round(float(t[1]), 3),
                }
            )
        except Exception as e:
            summary_rows.append(
                {
                    "Модель": m_name,
                    "Опис": f"Помилка / Відсутні ваги: {e}",
                    "2D Знімки": "-",
                    "Mean P(50km)": np.nan,
                    "Max P(50km)": np.nan,
                    "Low (%)": np.nan,
                    "Medium (%)": np.nan,
                    "High (%)": np.nan,
                    "High (к-сть сцен)": 0,
                    "Mean E[cnt 50km]": np.nan,
                    "Поріг Low/Med": np.nan,
                    "Поріг Med/High": np.nan,
                }
            )

    if model_a not in outputs:
        # Fallback to first available model
        if outputs:
            model_a = next(iter(outputs.keys()))
        else:
            raise gr.Error("Жодна з обраних моделей не змогла виконати прогноз. Перевірте вкладку 'Статус Моделей'.")
    if model_b not in outputs:
        model_b = model_a

    # Add diff column between Model A and Model B
    out_df["delta_P_A_minus_B"] = np.round(outputs[model_a]["p"] - outputs[model_b]["p"], 4)

    # Save predictions CSV to mini_dataset
    pred_csv_path = store.save_daily_predictions(day, out_df)

    # Render maps
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

    md_info = (
        f"### ✅ Розрахунок завершено для **{day}** ({len(pts)} локацій Іберії)\n"
        f"- **Порівняння `{model_a}` vs `{model_b}`**: збіг рівнів ризику = **{cmp_stats['level_agreement_pct']}%**, "
        f"кореляція Пірсона $r = {cmp_stats['pearson_r']}$, середня абсолютна різниця $|\\Delta P| = {cmp_stats['mean_abs_diff']}$.\n"
        f"- **Покриття FIRMS за останні 30 днів**: {grid_data['n_firms_detections']} детекцій за {grid_data['n_covered_days']}/30 днів | "
        f"**2D супутникових сцен у кеші за цю дату**: {grid_data['n_2d_scenes']}.\n"
        f"- **Збережено у міні-датасет**: `{pred_csv_path}`"
    )

    progress(1.0, desc="Готово!")
    return str(cmp_path), str(map_a_path), str(map_b_path), summary_df, top_df, str(pred_csv_path), md_info


# ==============================================================================
# Tab 2: Single Location + 22-Channel 2D Satellite Patch Visualization & Scoring
# ==============================================================================

def _render_2d_patch_Diagnostic(scene_2d: Dict[str, Any], lat: float, lon: float, day_str: str, out_path: Path) -> Path:
    """Renders an 8-panel visualization of the 22-channel 256x256 satellite & terrain cube."""
    x = np.asarray(scene_2d["X_2d"], dtype=np.float32)  # (22, 256, 256)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140)
    ax = axes.ravel()

    # 1. Sentinel-2 RGB True Color (B4=ch2, B3=ch1, B2=ch0)
    rgb = np.stack([x[2], x[1], x[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb_disp = np.clip(rgb / 0.30, 0.0, 1.0) ** 0.85
    ax[0].imshow(rgb_disp)
    ax[0].set_title(f"Sentinel-2 RGB ({scene_2d.get('s2_Source_Date', 'N/A')})")

    # 2. NDVI (ch10)
    im1 = ax[1].imshow(np.nan_to_num(x[10], 0.0), cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    ax[1].set_title("NDVI (Індекс рослинності)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    # 3. NBR (ch11)
    im2 = ax[2].imshow(np.nan_to_num(x[11], 0.0), cmap="Spectral", vmin=-0.5, vmax=0.7)
    ax[2].set_title("NBR (Normalized Burn Ratio)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    # 4. NDWI (ch12)
    im3 = ax[3].imshow(np.nan_to_num(x[12], 0.0), cmap="BrBG", vmin=-0.5, vmax=0.6)
    ax[3].set_title("NDWI (Вологість покриву)")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    # 5. Sentinel-1 VV (ch13)
    im4 = ax[4].imshow(np.nan_to_num(x[13], 0.0), cmap="gray")
    ax[4].set_title("Sentinel-1 SAR VV (Радар)")
    plt.colorbar(im4, ax=ax[4], fraction=0.046)

    # 6. DEM Elevation (ch16)
    im5 = ax[5].imshow(np.nan_to_num(x[16], 0.0), cmap="terrain")
    ax[5].set_title("Copernicus DEM Висота (м)")
    plt.colorbar(im5, ax=ax[5], fraction=0.046)

    # 7. Slope (ch17)
    im6 = ax[6].imshow(np.nan_to_num(x[17], 0.0), cmap="magma")
    ax[6].set_title("Нахил рельєфу (Slope, °)")
    plt.colorbar(im6, ax=ax[6], fraction=0.046)

    # 8. Dijkstra Road Travel Time (ch20)
    im7 = ax[7].imshow(np.nan_to_num(x[20], 0.0), cmap="viridis_r")
    ax[7].set_title("Час доїзду від доріг OSM (хв)")
    plt.colorbar(im7, ax=ax[7], fraction=0.046)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.suptitle(f"22-канальний 2D супутниковий патч (256×256 @ 10m) — ({lat:.4f}, {lon:.4f}) на {day_str}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _render_single_point_model_chart(preds: Dict[str, Any], lat: float, lon: float, day_str: str, out_path: Path) -> Path:
    """Renders a horizontal bar chart comparing P(fire within 50km) across all evaluated models."""
    valid = [(k, v) for k, v in preds.items() if isinstance(v, dict) and "P_fire_50km" in v]
    valid.sort(key=lambda item: item[1]["P_fire_50km"])

    fig, ax = plt.subplots(figsize=(9.5, max(3.8, 0.52 * len(valid))), dpi=140)
    names = [f"{k} {'(2D🛰️)' if v.get('used_2d_imagery') else '(1D)'}" for k, v in valid]
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
    ax.set_xlabel("Ймовірність пожежі в радіусі 50 км P(fire_50km)")
    ax.set_title(f"Порівняння моделей для ({lat:.4f}, {lon:.4f}) на {day_str}")
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
    progress(0.15, desc=f"Збір даних (1D погода, FIRMS історія, кліматологія, {'+ 2D супутниковий патч' if collect_2d else 'без 2D'})...")
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
                    "Модель": m,
                    "Тип": "2D Супутник + 1D" if v.get("used_2d_imagery") else "1D + Історія/Клімат",
                    "P(fire ≤ 50km)": v["P_fire_50km"],
                    "Рівень ризику": v["risk_level"],
                    "E[5km]": v["expected_count"]["5km"],
                    "E[10km]": v["expected_count"]["10km"],
                    "E[25km]": v["expected_count"]["25km"],
                    "E[50km]": v["expected_count"]["50km"],
                    "90% межа (50km)": v["upper_90pct_count"]["50km"],
                    "Пороги (Low/Med/High)": f"{v['thresholds_low_med_high'][0]:.2f} / {v['thresholds_low_med_high'][1]:.2f}",
                }
            )
    df_out = pd.DataFrame(rows).sort_values(by="P(fire ≤ 50km)", ascending=False).reset_index(drop=True)

    chart_path = store.maps_dir / f"point_models_{lat:.3f}_{lon:.3f}_{res['date']}.png"
    _render_single_point_model_chart(preds, float(lat), float(lon), res["date"], chart_path)

    patch_img_path = None
    if res.get("scene_2d") is not None:
        patch_img_path = store.maps_dir / f"patch2d_{lat:.3f}_{lon:.3f}_{res['date']}.png"
        _render_2d_patch_Diagnostic(res["scene_2d"], float(lat), float(lon), res["date"], patch_img_path)

    md = (
        f"### 📍 Результат для `({lat:.4f}, {lon:.4f})` на **{res['date']}**\n"
        f"- **2D супутниковий патч (22 канали)**: {'✅ Зібрано / Завантажено' if res['has_2d_scene'] else '⚪ Не збирався (тільки 1D моделі)'} "
        f"| **Дата знімка Sentinel-2**: `{res.get('s2_source_date')}`\n"
        f"- **Детекції FIRMS у радіусі 50 км**: за останні 7 днів = **{res['recent_fires_7d_50km']}**, за 30 днів = **{res['recent_fires_30d_50km']}**\n"
        f"- **Збережений `.npz` файл у міні-датасеті**: `{res.get('saved_npz_path') or '—'}`"
    )
    progress(1.0, desc="Готово!")
    return str(chart_path), (str(patch_img_path) if patch_img_path else None), df_out, md


# ==============================================================================
# Tab 3 & 4: Mini-Dataset Inspector, ZIP Export & System Status
# ==============================================================================

def refresh_mini_dataset_view():
    _, store = _get_registry_and_store()
    st = store.summary_stats()
    meta_df = pd.read_csv(store.metadata_path) if store.metadata_path.exists() else pd.DataFrame()
    md = (
        f"### 📦 Стан Міні-Датасету (`{st['root_dir']}`)\n"
        f"- **Зібрано 2D+1D супутникових сцен (`.npz`)**: `{st['num_npz_scenes']}`\n"
        f"- **Збережено денних таблиць ознак (`features_*.csv`)**: `{st['num_feature_days']}`\n"
        f"- **Збережено денних прогнозів моделей (`predictions_*.csv`)**: `{st['num_prediction_days']}`\n"
        f"- **Згенеровано карт (`.png`)**: `{st['num_saved_maps']}`\n"
        f"- **Загальний розмір**: `{st['total_size_mb']} MB`"
    )
    return md, meta_df


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
                "Код моделі": k,
                "Назва": st["label"],
                "Статус": "✅ Готова" if st["available"] else "⚠️ Відсутні файли ваг",
                "Потребує 2D (.npz)": "Так (22 канали)" if st["requires_2d"] else "Ні (1D + Історія)",
                "Відсутні файли": ", ".join(st["missing_files"]) if st["missing_files"] else "Усі файли на місці",
            }
        )
    df_models = pd.DataFrame(rows)

    data_checks = [
        ("metadata.csv (Сітка точок Іберії)", Path(_find_points_csv()).exists()),
        ("firms_covered_days.csv (Покриті дні FIRMS)", Path(CONFIG["HISTORICAL_COVERED_CSV"]).exists() or Path("data/firms_covered_days.csv").exists()),
        ("firms_points_2017_2025.csv.gz (Історичні пожежі 2017–2025)", Path(CONFIG["HISTORICAL_FIRMS_CSV"]).exists() or Path("data/firms_points_2017_2025.csv.gz").exists()),
        ("oof_predictions.csv (Калібрування порогів OOF)", reg.oof_path is not None and reg.oof_path.exists()),
        ("population_cache.csv (Кеш населення 10/50км без GEE)", Path(CONFIG["POPULATION_CACHE_CSV"]).exists()),
        ("FIRMS_MAP_KEY (Секрет для live FIRMS API)", bool(CONFIG.get("FIRMS_MAP_KEY"))),
    ]
    lines = ["### 🔍 Перевірка системних файлів та секретів Hugging Face Space"]
    for label, ok in data_checks:
        lines.append(f"- {'✅' if ok else '⚠️'} **{label}**")
    return "\n".join(lines), df_models


# ==============================================================================
# Build Gradio UI
# ==============================================================================

def build_app() -> gr.Blocks:
    choices = _model_choices()
    default_a = "tab_clim_full"
    default_b = "nn_ts" if any(k == "nn_ts" for _, k in choices) else "persist"

    with gr.Blocks(title="PreemptFire — Iberia Wildfire Risk & Model Comparison") as demo:
        gr.Markdown(
            "# 🔥 PreemptFire: Прогнозування Пожежної Небезпеки (Іберійський п-ів)\n"
            "Повноцінна система збору супутникових/метеорологічних даних (`Approach B`), візуального порівняння моделей "
            "(`1D`, `GRU Time-Series`, `2D SimCLR + CatBoost`) та автоматичного формування міні-датасету."
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # TAB 1: Map & Model Comparison
            # ------------------------------------------------------------------
            with gr.TabItem("🗺️ 1. Карта Іберії та Порівняння Моделей"):
                with gr.Row():
                    date_in = gr.Textbox(value="2025-08-12", label="Дата прогнозу (YYYY-MM-DD)", scale=1)
                    model_a_in = gr.Dropdown(choices=choices, value=default_a, label="Модель A (Основна карта)", scale=2)
                    model_b_in = gr.Dropdown(choices=choices, value=default_b, label="Модель B (Для порівняння)", scale=2)

                with gr.Accordion("🛰️ Налаштування 2D супутникових сцен (для tab_ssl_full та tab_full) і кешування", open=False):
                    with gr.Row():
                        also_all_in = gr.Checkbox(value=True, label="Одночасно порахувати метрики для ВСІХ доступних моделей у таблиці")
                        collect_2d_limit_in = gr.Slider(
                            minimum=0,
                            maximum=30,
                            step=1,
                            value=0,
                            label="Додатково зібрати N нових 2D супутникових сцен за цю дату (0 = використовувати вже кешовані)",
                        )
                        collect_2d_strat_in = gr.Radio(
                            choices=["top_risk", "random"],
                            value="top_risk",
                            label="Стратегія вибору точок для 2D збору",
                        )
                        force_refresh_in = gr.Checkbox(value=False, label="Примусово оновити кеш погоди/FIRMS")

                run_map_btn = gr.Button("🚀 Побудувати карти та порівняти моделі", variant="primary")
                map_status_md = gr.Markdown()

                gr.Markdown("### ⚖️ Візуальне порівняння: Модель A vs Модель B та Карта Різниць ($\\Delta P = P_A - P_B$)")
                cmp_img_out = gr.Image(label="Порівняння Модель A vs Модель B + ΔP", type="filepath")

                with gr.Row():
                    map_a_out = gr.Image(label="Детальна карта Моделі A (Рівень ризику + Очікувана к-сть пожеж 50км)", type="filepath")
                    map_b_out = gr.Image(label="Детальна карта Моделі B (Рівень ризику + Очікувана к-сть пожеж 50км)", type="filepath")

                gr.Markdown("### 📊 Зведена таблиця всіх моделей за обрану дату")
                summary_table_out = gr.Dataframe(label="Порівняння моделей на всій сітці Іберії")

                gr.Markdown("### 🔥 Топ-50 найнебезпечніших локацій")
                top_table_out = gr.Dataframe(label="Топ-50 точок за ймовірністю пожежі")
                csv_download_out = gr.File(label="Завантажити повний CSV прогнозів за день")

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
                        map_status_md,
                    ],
                )

            # ------------------------------------------------------------------
            # TAB 2: Live Single Point + 2D Satellite Patch (tab_ssl_full & tab_full)
            # ------------------------------------------------------------------
            with gr.TabItem("🛰️ 2. Точковий Аналіз + 2D Супутникові Знімки (tab_ssl_full / tab_full)"):
                gr.Markdown(
                    "Зберіть повний **22-канальний 2D супутниковий куб** (`Sentinel-2 L2A`, `Sentinel-1 SAR`, "
                    "`Copernicus DEM`, `OSM шляхи + час доїзду Дейкстри`, `ERA5/Open-Meteo вологість ґрунту`) "
                    "для будь-якої точки Іберії, збережіть її у міні-датасет і порівняйте **2D моделі (`tab_ssl_full`, `tab_full`)** з **1D моделями**."
                )
                with gr.Row():
                    pt_lat_in = gr.Number(value=40.15, label="Широта (lat, напр. 36.0 .. 43.8)")
                    pt_lon_in = gr.Number(value=-4.65, label="Довгота (lon, напр. -9.3 .. 3.2)")
                    pt_date_in = gr.Textbox(value="2025-08-12", label="Дата (YYYY-MM-DD)")
                    pt_2d_in = gr.Checkbox(
                        value=True,
                        label="🛰️ Зібрати 22-канальний 2D супутниковий патч (активує tab_ssl_full та tab_full)",
                    )

                run_pt_btn = gr.Button("🔍 Зібрати дані точки та порівняти всі 1D + 2D моделі", variant="primary")
                pt_status_md = gr.Markdown()

                with gr.Row():
                    pt_chart_out = gr.Image(label="Порівняння ймовірностей P(fire_50km) між усіма моделями", type="filepath")
                    pt_patch_out = gr.Image(label="Візуалізація 22-канального 2D патчу (RGB, NDVI, NBR, NDWI, SAR, DEM, Slope, Roads)", type="filepath")

                pt_table_out = gr.Dataframe(label="Детальна таблиця прогнозів (5км, 10км, 25км, 50км) по всіх моделях")

                run_pt_btn.click(
                    fn=run_single_point_analysis,
                    inputs=[pt_lat_in, pt_lon_in, pt_date_in, pt_2d_in],
                    outputs=[pt_chart_out, pt_patch_out, pt_table_out, pt_status_md],
                )

            # ------------------------------------------------------------------
            # TAB 3: Mini-Dataset Store & Export
            # ------------------------------------------------------------------
            with gr.TabItem("📦 3. Міні-Датасет (Зібрані Дані та Результати)"):
                gr.Markdown(
                    "Усі зібрані 2D супутникові сцени (`.npz`), таблиці ознак (`features_*.csv`), прогнози моделей (`predictions_*.csv`) "
                    "та карти автоматично зберігаються у папці `mini_dataset/` у форматі вашого репозиторію `approach_b`."
                )
                with gr.Row():
                    refresh_ds_btn = gr.Button("🔄 Оновити статистику міні-датасету")
                    export_zip_btn = gr.Button("📥 Запакувати та завантажити весь міні-датасет (.zip)", variant="primary")

                ds_status_md = gr.Markdown()
                zip_file_out = gr.File(label="Архів міні-датасету (.zip)")
                ds_meta_table = gr.Dataframe(label="Зібрані 2D/1D сцени (mini_dataset/metadata.csv)")

                refresh_ds_btn.click(fn=refresh_mini_dataset_view, inputs=[], outputs=[ds_status_md, ds_meta_table])
                export_zip_btn.click(fn=create_mini_dataset_zip, inputs=[], outputs=[zip_file_out])

            # ------------------------------------------------------------------
            # TAB 4: Model & File Diagnostics
            # ------------------------------------------------------------------
            with gr.TabItem("⚙️ 4. Статус Моделей та Файлів"):
                check_btn = gr.Button("🔄 Перевірити наявність моделей та файлів даних")
                sys_md, sys_df = get_system_status_markdown()
                sys_md_out = gr.Markdown(value=sys_md)
                sys_df_out = gr.Dataframe(value=sys_df, label="Статус усіх 11 моделей у реєстрі")
                check_btn.click(fn=get_system_status_markdown, inputs=[], outputs=[sys_md_out, sys_df_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
