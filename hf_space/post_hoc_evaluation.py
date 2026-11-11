"""Independent post-hoc ground-truth fire detection verification module.

This module is strictly isolated from feature engineering (`config_and_features.py`),
live data collection for inference (`live.py`, `make_map.py`), and model prediction
(`model_registry.py`). It is only invoked AFTER predictions have been generated
using strictly past data (up to target_day - 1), preventing any data leakage.

It fetches actual NASA FIRMS VIIRS fire detections for:
  1. The exact target day: [target_day, target_day] (1 day)
  2. The 7-day forward window: [target_day, target_day + 6 days] (7 days)
and evaluates how well the predicted risk levels (Medium and High separately,
as well as combined Medium+High) cover all grid locations within radius R of actual fires.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config_and_features import IBERIA_BBOX, RADII, R_EARTH, dist_km, to_xyz
from data_collectors import fetch_firms_days


def fetch_post_hoc_firms(
    target_day: date,
    bbox: str = IBERIA_BBOX,
    progress_cb=None,
) -> Dict[str, Any]:
    """Independently fetches actual FIRMS detections for target_day (1d) and [target_day, target_day + 6d] (7d)."""
    end_7d = target_day + timedelta(days=6)
    if progress_cb:
        progress_cb(
            0.15,
            f"[Post-Hoc] Fetching actual NASA FIRMS detections for {target_day} and 7-day window ({target_day} to {end_7d})...",
        )

    fire_7d_df, covered_7d = fetch_firms_days(
        target_day,
        end_7d,
        bbox=bbox,
        progress_cb=None,
    )

    if len(fire_7d_df) > 0:
        fire_7d_df = fire_7d_df.copy()
        fire_7d_df["acq_date_str"] = fire_7d_df["acq_date"].astype(str).str[:10]
        fire_1d_df = fire_7d_df[fire_7d_df["acq_date_str"] == target_day.isoformat()].copy().reset_index(drop=True)
    else:
        fire_7d_df = pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
        fire_1d_df = pd.DataFrame(columns=["latitude", "longitude", "acq_date"])

    covered_1d = {target_day} if target_day in covered_7d else set()

    return {
        "target_day": target_day,
        "end_7d": end_7d,
        "fire_1d_df": fire_1d_df,
        "fire_7d_df": fire_7d_df,
        "covered_1d": covered_1d,
        "covered_7d": covered_7d,
    }


def compute_actual_counts_per_point(
    pts: pd.DataFrame,
    fire_df: pd.DataFrame,
    radii: Tuple[int, ...] = RADII,
) -> Dict[int, np.ndarray]:
    """Computes the number of actual fire detections within each radius R (km) for every grid point."""
    n = len(pts)
    counts = {int(R): np.zeros(n, dtype=np.int32) for R in radii}
    if len(fire_df) == 0 or n == 0:
        return counts

    pts_xyz = to_xyz(pts["lat"].values, pts["lon"].values)
    f_xyz = to_xyz(fire_df["latitude"].values, fire_df["longitude"].values)
    max_r = max(radii)
    ch_max = 2.0 * np.sin(max_r / (2.0 * R_EARTH))

    tree = cKDTree(f_xyz)
    neighbors = tree.query_ball_point(pts_xyz, r=ch_max)
    for idx, lst in enumerate(neighbors):
        if not lst:
            continue
        dk = dist_km(tree.data[lst], pts_xyz[idx])
        for R in radii:
            counts[int(R)][idx] = int((dk <= R).sum())

    return counts


def evaluate_levels_against_actual(
    levels: np.ndarray,
    probs: np.ndarray,
    actual_counts: np.ndarray,
    model_name: str,
    window_label: str,
    radius_km: int,
) -> Dict[str, Any]:
    """Evaluates how grid points with actual fires (actual_counts > 0) are classified into High, Medium, and Low."""
    levels_arr = np.asarray(levels, dtype=str)
    probs_arr = np.asarray(probs, dtype=np.float64)
    act_arr = np.asarray(actual_counts, dtype=np.int32)

    has_fire = act_arr > 0
    n_total = len(act_arr)
    n_fire = int(has_fire.sum())

    is_high = levels_arr == "High"
    is_med = levels_arr == "Medium"
    is_low = levels_arr == "Low"

    fire_high = int((has_fire & is_high).sum())
    fire_med = int((has_fire & is_med).sum())
    fire_low = int((has_fire & is_low).sum())
    fire_med_or_high = fire_high + fire_med

    if n_fire > 0:
        pct_high = round(100.0 * fire_high / n_fire, 1)
        pct_med = round(100.0 * fire_med / n_fire, 1)
        pct_med_high = round(100.0 * fire_med_or_high / n_fire, 1)
        pct_low_missed = round(100.0 * fire_low / n_fire, 1)
        all_high = bool(fire_high == n_fire)
        all_med = bool(fire_med == n_fire)
        all_med_or_high = bool(fire_med_or_high == n_fire)
    else:
        pct_high = np.nan
        pct_med = np.nan
        pct_med_high = np.nan
        pct_low_missed = np.nan
        all_high = False
        all_med = False
        all_med_or_high = False

    n_pred_high = int(is_high.sum())
    n_pred_med = int(is_med.sum())
    prec_high = round(100.0 * fire_high / n_pred_high, 1) if n_pred_high > 0 else np.nan
    prec_med = round(100.0 * fire_med / n_pred_med, 1) if n_pred_med > 0 else np.nan

    mean_p_fire = round(float(probs_arr[has_fire].mean()), 4) if n_fire > 0 else np.nan
    mean_p_nofire = round(float(probs_arr[~has_fire].mean()), 4) if (n_total - n_fire) > 0 else np.nan

    return {
        "Model": model_name,
        "Window": window_label,
        "Radius (km)": int(radius_km),
        "Active Fire Scenes": n_fire,
        "Marked High (count)": fire_high,
        "Marked High (%)": pct_high,
        "All in Radius High?": "Yes" if all_high else ("No fires" if n_fire == 0 else "No"),
        "Marked Medium (count)": fire_med,
        "Marked Medium (%)": pct_med,
        "All in Radius Medium?": "Yes" if all_med else ("No fires" if n_fire == 0 else "No"),
        "Marked Med+High (count)": fire_med_or_high,
        "Marked Med+High (%)": pct_med_high,
        "All in Radius Med+High?": "Yes" if all_med_or_high else ("No fires" if n_fire == 0 else "No"),
        "Missed / Low (count)": fire_low,
        "Missed / Low (%)": pct_low_missed,
        "Precision High (%)": prec_high,
        "Precision Medium (%)": prec_med,
        "Mean P (Fire Scenes)": mean_p_fire,
        "Mean P (No-Fire Scenes)": mean_p_nofire,
    }


def evaluate_local_radius_around_point(
    pts: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    search_radius_km: float,
    levels: np.ndarray,
    probs: np.ndarray,
    actual_1d: np.ndarray,
    actual_7d: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Checks all grid scenes located within `search_radius_km` of `(center_lat, center_lon)`
    and reports whether all of them are marked Medium and High (separately), plus their actual fire status.
    """
    pts_xyz = to_xyz(pts["lat"].values, pts["lon"].values)
    c_xyz = to_xyz([center_lat], [center_lon])[0]
    dk = dist_km(pts_xyz, c_xyz)
    in_rad = dk <= float(search_radius_km)

    n_in = int(in_rad.sum())
    if n_in == 0:
        return {
            "Model": model_name,
            "Radius (km)": search_radius_km,
            "Scenes in Radius": 0,
            "High (count)": 0,
            "High (%)": np.nan,
            "All High?": "No scenes",
            "Medium (count)": 0,
            "Medium (%)": np.nan,
            "All Medium?": "No scenes",
            "Med+High (count)": 0,
            "Med+High (%)": np.nan,
            "All Med+High?": "No scenes",
            "Low (count)": 0,
            "Actual Fire 1d Scenes": 0,
            "Actual Fire 7d Scenes": 0,
        }

    lev_sub = np.asarray(levels, dtype=str)[in_rad]
    n_high = int((lev_sub == "High").sum())
    n_med = int((lev_sub == "Medium").sum())
    n_low = int((lev_sub == "Low").sum())
    n_mh = n_high + n_med

    return {
        "Model": model_name,
        "Radius (km)": search_radius_km,
        "Scenes in Radius": n_in,
        "High (count)": n_high,
        "High (%)": round(100.0 * n_high / n_in, 1),
        "All High?": "Yes" if n_high == n_in else "No",
        "Medium (count)": n_med,
        "Medium (%)": round(100.0 * n_med / n_in, 1),
        "All Medium?": "Yes" if n_med == n_in else "No",
        "Med+High (count)": n_mh,
        "Med+High (%)": round(100.0 * n_mh / n_in, 1),
        "All Med+High?": "Yes" if n_mh == n_in else "No",
        "Low (count)": n_low,
        "Actual Fire 1d Scenes": int((actual_1d[in_rad] > 0).sum()),
        "Actual Fire 7d Scenes": int((actual_7d[in_rad] > 0).sum()),
    }


def draw_ground_truth_comparison_map(
    pts: pd.DataFrame,
    levels: np.ndarray,
    probs: np.ndarray,
    thresholds: Tuple[float, float],
    model_name: str,
    target_day: date,
    fire_1d_df: pd.DataFrame,
    fire_7d_df: pd.DataFrame,
    actual_1d_r: np.ndarray,
    actual_7d_r: np.ndarray,
    radius_km: int,
    out_path: str | Path,
) -> Path:
    """Renders a 3-panel post-hoc comparison map:
      Panel 1: Predicted Risk Map (Low / Medium / High) + raw VIIRS detections on Target Day
      Panel 2: Target Day (1d) Ground-Truth Verification (High hit, Medium hit, Low missed, No fire)
      Panel 3: 7-Day Forward Window (7d) Ground-Truth Verification (High hit, Medium hit, Low missed, No fire)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels_arr = np.asarray(levels, dtype=str)
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5), dpi=150)

    level_colors = {"Low": "#fee08b", "Medium": "#f46d43", "High": "#a50026"}
    for lv_name in ("Low", "Medium", "High"):
        m = levels_arr == lv_name
        ax[0].scatter(
            pts["lon"][m],
            pts["lat"][m],
            c=level_colors[lv_name],
            s=14,
            label=f"Pred {lv_name}: {int(m.sum())} ({100.0 * m.mean():.0f}%)",
            edgecolors="none",
        )
    if len(fire_1d_df) > 0:
        ax[0].scatter(
            fire_1d_df["longitude"],
            fire_1d_df["latitude"],
            c="#00e5ff",
            marker="*",
            s=55,
            edgecolors="black",
            linewidths=0.5,
            label=f"VIIRS detections on {target_day} ({len(fire_1d_df)})",
            zorder=5,
        )
    ax[0].set_title(
        f"[{model_name}] Predicted Risk on {target_day}\n"
        f"Thresholds: {thresholds[0]:.2f} / {thresholds[1]:.2f} | Mean P={np.mean(probs):.3f}"
    )
    ax[0].set_xlabel("Longitude")
    ax[0].set_ylabel("Latitude")
    ax[0].legend(loc="lower left", frameon=True, fontsize=8)
    ax[0].grid(alpha=0.25)

    windows = [
        (
            1,
            f"Target Day ({target_day})",
            actual_1d_r,
            fire_1d_df,
        ),
        (
            2,
            f"7-Day Window ({target_day} to {target_day + timedelta(days=6)})",
            actual_7d_r,
            fire_7d_df,
        ),
    ]

    for panel_idx, title_tag, act_cnt, f_df in windows:
        has_f = act_cnt > 0
        no_f = ~has_f
        hit_high = has_f & (levels_arr == "High")
        hit_med = has_f & (levels_arr == "Medium")
        miss_low = has_f & (levels_arr == "Low")
        n_f = int(has_f.sum())

        ax[panel_idx].scatter(
            pts["lon"][no_f],
            pts["lat"][no_f],
            c="#e0e0e0",
            s=10,
            label=f"No fire <= {radius_km}km ({int(no_f.sum())})",
            edgecolors="none",
            alpha=0.7,
        )
        if hit_med.any():
            pct_m = 100.0 * int(hit_med.sum()) / max(n_f, 1)
            ax[panel_idx].scatter(
                pts["lon"][hit_med],
                pts["lat"][hit_med],
                c="#f46d43",
                s=28,
                label=f"Fire + Medium: {int(hit_med.sum())}/{n_f} ({pct_m:.1f}%)",
                edgecolors="#222222",
                linewidths=0.4,
                zorder=3,
            )
        else:
            ax[panel_idx].scatter([], [], c="#f46d43", s=28, label=f"Fire + Medium: 0/{n_f} (0.0%)")

        if hit_high.any():
            pct_h = 100.0 * int(hit_high.sum()) / max(n_f, 1)
            ax[panel_idx].scatter(
                pts["lon"][hit_high],
                pts["lat"][hit_high],
                c="#a50026",
                s=32,
                label=f"Fire + High: {int(hit_high.sum())}/{n_f} ({pct_h:.1f}%)",
                edgecolors="#111111",
                linewidths=0.5,
                zorder=4,
            )
        else:
            ax[panel_idx].scatter([], [], c="#a50026", s=32, label=f"Fire + High: 0/{n_f} (0.0%)")

        if miss_low.any():
            pct_l = 100.0 * int(miss_low.sum()) / max(n_f, 1)
            ax[panel_idx].scatter(
                pts["lon"][miss_low],
                pts["lat"][miss_low],
                c="#2b83ba",
                marker="^",
                s=38,
                label=f"MISSED (Fire + Low): {int(miss_low.sum())}/{n_f} ({pct_l:.1f}%)",
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        else:
            ax[panel_idx].scatter([], [], c="#2b83ba", marker="^", s=38, label=f"MISSED (Fire + Low): 0/{n_f} (0.0%)")

        if len(f_df) > 0:
            ax[panel_idx].scatter(
                f_df["longitude"],
                f_df["latitude"],
                c="#00e5ff",
                marker="*",
                s=40,
                edgecolors="black",
                linewidths=0.4,
                label=f"Raw VIIRS points ({len(f_df)})",
                zorder=6,
            )

        pct_mh = 100.0 * int((hit_high | hit_med).sum()) / max(n_f, 1) if n_f > 0 else 0.0
        ax[panel_idx].set_title(
            f"Ground Truth: {title_tag} (R={radius_km}km)\n"
            f"Active scenes: {n_f} | Captured by Med+High: {pct_mh:.1f}% (High: {int(hit_high.sum())}, Med: {int(hit_med.sum())})"
        )
        ax[panel_idx].set_xlabel("Longitude")
        ax[panel_idx].set_ylabel("Latitude")
        ax[panel_idx].legend(loc="lower left", frameon=True, fontsize=8)
        ax[panel_idx].grid(alpha=0.25)

    plt.tight_layout()
    out_path = Path(out_path)
    plt.savefig(out_path)
    plt.close(fig)
    return out_path
