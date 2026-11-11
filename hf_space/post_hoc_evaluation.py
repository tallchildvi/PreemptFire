"""Independent post-hoc ground-truth fire detection verification module.

This module is strictly isolated from feature engineering (`config_and_features.py`),
live data collection for inference (`live.py`, `make_map.py`), and model prediction
(`model_registry.py`). It is only invoked AFTER predictions have been generated
using strictly past data (up to target_day - 1), preventing any data leakage.

Because wildfire ignition is a stochastic process, evaluation focuses on probabilistic
calibration and risk stratification:
  - Empirical fire rate per predicted risk tier: P(fire | Low), P(fire | Medium), P(fire | High)
  - Monotonicity and Risk Ratio: P(fire | High) / P(fire | Low)
  - Lift over baseline prevalence: P(fire | tier) / P(fire overall)
  - Threshold-independent probabilistic metrics: Brier Score, ROC-AUC, PR-AUC
  - Low-tier safety rate: 1 - P(fire | Low) and share of fires missed in Low
  - Multi-date batch evaluation for seasonal/aggregate reliability
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


def _roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC via rank statistics (Mann-Whitney U) in pure NumPy."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(ranks[y].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _pr_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Average Precision (PR-AUC) in pure NumPy."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted).astype(np.float64)
    k = np.arange(1, len(y) + 1, dtype=np.float64)
    precision_at_k = tp / k
    ap = float(precision_at_k[y_sorted].sum() / n_pos)
    return float(np.clip(ap, 0.0, 1.0))


def _brier_score_np(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier Score mean((p - y)^2)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if len(y) == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))


def fetch_post_hoc_firms(
    target_day: date,
    bbox: str = IBERIA_BBOX,
    firms_keys: Optional[str | Sequence[str]] = None,
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
        firms_keys=firms_keys,
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

    from data_collectors import _LAST_FIRMS_DIAGNOSTICS
    diag = dict(_LAST_FIRMS_DIAGNOSTICS)
    diag["covered_1d_ok"] = bool(len(covered_1d) > 0)
    diag["covered_7d_count"] = len(covered_7d)

    return {
        "target_day": target_day,
        "end_7d": end_7d,
        "fire_1d_df": fire_1d_df,
        "fire_7d_df": fire_7d_df,
        "covered_1d": covered_1d,
        "covered_7d": covered_7d,
        "status_info": diag,
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


def evaluate_tier_reliability_table(
    levels: np.ndarray,
    probs: np.ndarray,
    actual_counts: np.ndarray,
    model_name: str,
    window_label: str,
    radius_km: int,
) -> pd.DataFrame:
    """Builds a tier-by-tier reliability & lift table for a single model and time window:
    Rows: Low, Medium, High, Medium+High (Combined), Total (Base Rate).
    """
    levels_arr = np.asarray(levels, dtype=str)
    probs_arr = np.asarray(probs, dtype=np.float64)
    has_fire = np.asarray(actual_counts, dtype=np.int32) > 0

    n_total = len(has_fire)
    n_fire_total = int(has_fire.sum())
    base_rate = (n_fire_total / n_total) if n_total > 0 else 0.0

    tiers = [
        ("Low", levels_arr == "Low"),
        ("Medium", levels_arr == "Medium"),
        ("High", levels_arr == "High"),
        ("Medium + High", (levels_arr == "Medium") | (levels_arr == "High")),
        ("All Grid (Baseline)", np.ones(n_total, dtype=bool)),
    ]

    rows = []
    for tier_name, mask in tiers:
        n_scenes = int(mask.sum())
        area_share_pct = round(100.0 * n_scenes / max(n_total, 1), 1)
        n_f = int((has_fire & mask).sum())
        fire_rate = (n_f / n_scenes) if n_scenes > 0 else np.nan
        fire_rate_pct = round(100.0 * fire_rate, 1) if np.isfinite(fire_rate) else np.nan
        mean_pred_p = round(100.0 * float(probs_arr[mask].mean()), 1) if n_scenes > 0 else np.nan
        capture_recall_pct = round(100.0 * n_f / n_fire_total, 1) if n_fire_total > 0 else np.nan
        lift = round(fire_rate / base_rate, 2) if (np.isfinite(fire_rate) and base_rate > 1e-9) else np.nan

        rows.append(
            {
                "Model": model_name,
                "Window": window_label,
                "Radius (km)": int(radius_km),
                "Risk Tier": tier_name,
                "Grid Scenes": n_scenes,
                "Map Share (%)": area_share_pct,
                "Scenes with Fire": n_f,
                "Actual Fire Rate P(fire|Tier) (%)": fire_rate_pct,
                "Mean Predicted P (%)": mean_pred_p,
                "Lift vs Baseline (x)": lift,
                "Share of Total Fires Captured (%)": capture_recall_pct,
            }
        )

    return pd.DataFrame(rows)


def evaluate_levels_against_actual(
    levels: np.ndarray,
    probs: np.ndarray,
    actual_counts: np.ndarray,
    model_name: str,
    window_label: str,
    radius_km: int,
) -> Dict[str, Any]:
    """Computes probabilistic calibration, reliability by tier, lift, Brier score, ROC-AUC, and PR-AUC."""
    levels_arr = np.asarray(levels, dtype=str)
    probs_arr = np.asarray(probs, dtype=np.float64)
    act_arr = np.asarray(actual_counts, dtype=np.int32)

    has_fire = act_arr > 0
    n_total = len(act_arr)
    n_fire = int(has_fire.sum())
    base_rate = (n_fire / n_total) if n_total > 0 else 0.0

    is_high = levels_arr == "High"
    is_med = levels_arr == "Medium"
    is_low = levels_arr == "Low"
    is_mh = is_high | is_med

    n_high = int(is_high.sum())
    n_med = int(is_med.sum())
    n_low = int(is_low.sum())
    n_mh = int(is_mh.sum())

    fire_high = int((has_fire & is_high).sum())
    fire_med = int((has_fire & is_med).sum())
    fire_low = int((has_fire & is_low).sum())
    fire_mh = fire_high + fire_med

    rate_low = (fire_low / n_low) if n_low > 0 else np.nan
    rate_med = (fire_med / n_med) if n_med > 0 else np.nan
    rate_high = (fire_high / n_high) if n_high > 0 else np.nan
    rate_mh = (fire_mh / n_mh) if n_mh > 0 else np.nan

    lift_high = round(rate_high / base_rate, 2) if (np.isfinite(rate_high) and base_rate > 1e-9) else np.nan
    lift_mh = round(rate_mh / base_rate, 2) if (np.isfinite(rate_mh) and base_rate > 1e-9) else np.nan
    ratio_high_vs_low = round(rate_high / max(rate_low, 1e-4), 1) if (np.isfinite(rate_high) and np.isfinite(rate_low) and rate_low > 0) else np.nan

    monotonic = bool(
        (not np.isfinite(rate_low) or not np.isfinite(rate_med) or rate_low <= rate_med + 1e-6)
        and (not np.isfinite(rate_med) or not np.isfinite(rate_high) or rate_med <= rate_high + 1e-6)
    )

    recall_high = round(100.0 * fire_high / n_fire, 1) if n_fire > 0 else np.nan
    recall_med = round(100.0 * fire_med / n_fire, 1) if n_fire > 0 else np.nan
    recall_mh = round(100.0 * fire_mh / n_fire, 1) if n_fire > 0 else np.nan
    missed_low_share = round(100.0 * fire_low / n_fire, 1) if n_fire > 0 else np.nan

    brier = _brier_score_np(has_fire, probs_arr)
    roc_auc = _roc_auc_np(has_fire, probs_arr)
    pr_auc = _pr_auc_np(has_fire, probs_arr)

    return {
        "Model": model_name,
        "Window": window_label,
        "Radius (km)": int(radius_km),
        "Base Fire Rate (%)": round(100.0 * base_rate, 1),
        "P(fire | Low) (%)": round(100.0 * rate_low, 1) if np.isfinite(rate_low) else np.nan,
        "P(fire | Med) (%)": round(100.0 * rate_med, 1) if np.isfinite(rate_med) else np.nan,
        "P(fire | High) (%)": round(100.0 * rate_high, 1) if np.isfinite(rate_high) else np.nan,
        "High/Low Rate Ratio (x)": ratio_high_vs_low,
        "Monotonic (Low<=Med<=High)": "Yes" if monotonic else "No",
        "Lift High (x)": lift_high,
        "Lift Med+High (x)": lift_mh,
        "Recall Med+High (%)": recall_mh,
        "Recall High (%)": recall_high,
        "Recall Med (%)": recall_med,
        "Missed in Low (%)": missed_low_share,
        "Low Safe Rate (%)": round(100.0 * (1.0 - rate_low), 1) if np.isfinite(rate_low) else np.nan,
        "ROC-AUC": round(roc_auc, 4) if np.isfinite(roc_auc) else np.nan,
        "PR-AUC": round(pr_auc, 4) if np.isfinite(pr_auc) else np.nan,
        "Brier Score": round(brier, 4) if np.isfinite(brier) else np.nan,
    }


def evaluate_vs_fwi_benchmark(df_models: pd.DataFrame) -> pd.DataFrame:
    """Computes direct head-to-head comparison of every evaluated model against the
    Canadian Fire Weather Index (`fwi` / EFFIS standard) across each window (1d and 7d).
    """
    if df_models is None or len(df_models) == 0 or "Model" not in df_models.columns:
        return pd.DataFrame()

    df = df_models.copy()
    fwi_rows = df[df["Model"] == "fwi"]
    if len(fwi_rows) == 0:
        return df

    out_rows = []
    for win, grp in df.groupby("Window", sort=False):
        fwi_sub = grp[grp["Model"] == "fwi"]
        if len(fwi_sub) == 0:
            continue
        f_row = fwi_sub.iloc[0]
        f_auc = float(f_row.get("ROC-AUC", np.nan))
        f_pr = float(f_row.get("PR-AUC", np.nan))
        f_ratio = float(f_row.get("High/Low Rate Ratio (x)", np.nan))
        f_lift = float(f_row.get("Lift High (x)", np.nan))

        for _, r in grp.iterrows():
            m_auc = float(r.get("ROC-AUC", np.nan))
            m_pr = float(r.get("PR-AUC", np.nan))
            m_ratio = float(r.get("High/Low Rate Ratio (x)", np.nan))
            m_lift = float(r.get("Lift High (x)", np.nan))

            d_auc = round(m_auc - f_auc, 4) if (np.isfinite(m_auc) and np.isfinite(f_auc)) else np.nan
            d_pr = round(m_pr - f_pr, 4) if (np.isfinite(m_pr) and np.isfinite(f_pr)) else np.nan
            d_ratio = round(m_ratio - f_ratio, 2) if (np.isfinite(m_ratio) and np.isfinite(f_ratio)) else np.nan
            d_lift = round(m_lift - f_lift, 2) if (np.isfinite(m_lift) and np.isfinite(f_lift)) else np.nan

            # Relative skill over random (AUC - 0.5) compared to FWI (FWI_AUC - 0.5)
            if np.isfinite(m_auc) and np.isfinite(f_auc) and (f_auc - 0.5) > 1e-4:
                skill_gain_pct = round(100.0 * ((m_auc - 0.5) - (f_auc - 0.5)) / (f_auc - 0.5), 1)
            else:
                skill_gain_pct = np.nan

            row_dict = dict(r)
            row_dict["Delta ROC-AUC vs FWI"] = d_auc
            row_dict["Skill Gain vs FWI (%)"] = skill_gain_pct
            row_dict["Delta PR-AUC vs FWI"] = d_pr
            row_dict["Delta High/Low Ratio vs FWI"] = d_ratio
            row_dict["Delta Lift High vs FWI"] = d_lift
            out_rows.append(row_dict)

    return pd.DataFrame(out_rows)


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
    """Summarizes predicted risk distribution and empirical fire rates within `search_radius_km` of `(center_lat, center_lon)`."""
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
            "Mean P (%)": np.nan,
            "Low Scenes": 0,
            "Med Scenes": 0,
            "High Scenes": 0,
            "Med+High Share (%)": np.nan,
            "Actual Fire 1d Scenes": 0,
            "Actual Fire 1d Rate (%)": np.nan,
            "Actual Fire 7d Scenes": 0,
            "Actual Fire 7d Rate (%)": np.nan,
        }

    lev_sub = np.asarray(levels, dtype=str)[in_rad]
    p_sub = np.asarray(probs, dtype=np.float64)[in_rad]
    f1_sub = np.asarray(actual_1d, dtype=np.int32)[in_rad] > 0
    f7_sub = np.asarray(actual_7d, dtype=np.int32)[in_rad] > 0

    n_high = int((lev_sub == "High").sum())
    n_med = int((lev_sub == "Medium").sum())
    n_low = int((lev_sub == "Low").sum())
    n_mh = n_high + n_med

    return {
        "Model": model_name,
        "Radius (km)": search_radius_km,
        "Scenes in Radius": n_in,
        "Mean P (%)": round(100.0 * float(p_sub.mean()), 1),
        "Low Scenes": n_low,
        "Med Scenes": n_med,
        "High Scenes": n_high,
        "Med+High Share (%)": round(100.0 * n_mh / n_in, 1),
        "Actual Fire 1d Scenes": int(f1_sub.sum()),
        "Actual Fire 1d Rate (%)": round(100.0 * float(f1_sub.mean()), 1),
        "Actual Fire 7d Scenes": int(f7_sub.sum()),
        "Actual Fire 7d Rate (%)": round(100.0 * float(f7_sub.mean()), 1),
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
    """Renders a 3-panel post-hoc comparison map + reliability subtitle:
      Panel 1: Predicted Risk Map (Low / Medium / High) + raw VIIRS detections on Target Day
      Panel 2: Target Day (1d) Ground-Truth Verification with empirical fire rate per tier
      Panel 3: 7-Day Forward Window (7d) Ground-Truth Verification with empirical fire rate per tier
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels_arr = np.asarray(levels, dtype=str)
    probs_arr = np.asarray(probs, dtype=np.float64)
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.6), dpi=150)

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
        f"[{model_name}] Predicted Risk Field on {target_day}\n"
        f"Thresholds: {thresholds[0]:.2f} / {thresholds[1]:.2f} | Mean P={probs_arr.mean():.3f}"
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
        is_h = levels_arr == "High"
        is_m = levels_arr == "Medium"
        is_l = levels_arr == "Low"

        hit_high = has_f & is_h
        hit_med = has_f & is_m
        miss_low = has_f & is_l
        n_f = int(has_f.sum())

        r_l = 100.0 * int(miss_low.sum()) / max(int(is_l.sum()), 1)
        r_m = 100.0 * int(hit_med.sum()) / max(int(is_m.sum()), 1)
        r_h = 100.0 * int(hit_high.sum()) / max(int(is_h.sum()), 1)
        auc_val = _roc_auc_np(has_f, probs_arr)
        auc_str = f"{auc_val:.3f}" if np.isfinite(auc_val) else "N/A"

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
            ax[panel_idx].scatter(
                pts["lon"][hit_med],
                pts["lat"][hit_med],
                c="#f46d43",
                s=28,
                label=f"Fire in Medium: {int(hit_med.sum())}/{int(is_m.sum())} (P(fire|Med)={r_m:.1f}%)",
                edgecolors="#222222",
                linewidths=0.4,
                zorder=3,
            )
        else:
            ax[panel_idx].scatter([], [], c="#f46d43", s=28, label=f"Fire in Medium: 0/{int(is_m.sum())} (0.0%)")

        if hit_high.any():
            ax[panel_idx].scatter(
                pts["lon"][hit_high],
                pts["lat"][hit_high],
                c="#a50026",
                s=32,
                label=f"Fire in High: {int(hit_high.sum())}/{int(is_h.sum())} (P(fire|High)={r_h:.1f}%)",
                edgecolors="#111111",
                linewidths=0.5,
                zorder=4,
            )
        else:
            ax[panel_idx].scatter([], [], c="#a50026", s=32, label=f"Fire in High: 0/{int(is_h.sum())} (0.0%)")

        if miss_low.any():
            ax[panel_idx].scatter(
                pts["lon"][miss_low],
                pts["lat"][miss_low],
                c="#2b83ba",
                marker="^",
                s=38,
                label=f"Fire in Low: {int(miss_low.sum())}/{int(is_l.sum())} (P(fire|Low)={r_l:.1f}%)",
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        else:
            ax[panel_idx].scatter([], [], c="#2b83ba", marker="^", s=38, label=f"Fire in Low: 0/{int(is_l.sum())} (0.0%)")

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
            f"{title_tag} (R={radius_km}km) | ROC-AUC={auc_str} | Recall(Med+High)={pct_mh:.1f}%\n"
            f"Fire Rate by Tier: Low {r_l:.1f}% -> Med {r_m:.1f}% -> High {r_h:.1f}%"
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
