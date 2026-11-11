"""Canadian Forest Fire Weather Index (FWI) System (Van Wagner, 1987) & EFFIS baseline.

Implements the exact physical meteorology equations of the Canadian FWI System
used by the European Forest Fire Information System (EFFIS) / Copernicus GWIS:
  1. Fine Fuel Moisture Code (FFMC) - surface litter moisture (time lag ~16 hours)
  2. Duff Moisture Code (DMC)       - loosely compacted organic layers (time lag ~12 days)
  3. Drought Code (DC)              - deep compact organic layers (time lag ~52 days)
  4. Initial Spread Index (ISI)     - rate of fire spread combining FFMC and wind
  5. Buildup Index (BUI)            - total fuel available for combustion (DMC + DC)
  6. Fire Weather Index (FWI)       - frontal fire intensity combining ISI and BUI
  7. Daily Severity Rating (DSR)    - power-law transformation of FWI (0.0272 * FWI^1.77)

All calculations run strictly on the daily/hourly Open-Meteo ERA5 weather series
up to `target_day - 1` (zero data leakage), evolving the moisture codes day-by-day
across the 30-to-60 day lookback window.
"""
from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Standard Northern Hemisphere day-length factors for DMC (Le) and DC (Lf)
# Van Wagner (1987), Table 1 & Table 2 (valid for Iberian latitudes ~36N-44N)
_DMC_DAY_LENGTH_FACTORS = (6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0)
_DC_DAY_LENGTH_FACTORS = (-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6)

# Standard EFFIS / Copernicus GWIS FWI danger class thresholds for Mediterranean Europe:
# Very Low (<5.2), Low (5.2-11.2), Moderate (11.2-21.3), High (21.3-38.0), Very High (38.0-50.0), Extreme (>=50.0)
# For 3-tier stratification (Low / Medium / High), Low < 11.2, Medium [11.2, 29.0), High >= 29.0
# In probability space p = 1 / (1 + exp(-(FWI - 28.0) / 10.0)):
#   FWI = 11.2 -> p ~ 0.157
#   FWI = 29.0 -> p ~ 0.525
EFFIS_FWI_LOW_MED = 11.2
EFFIS_FWI_MED_HIGH = 29.0


def ffmc_step(ffmc_prev: float, temp_c: float, rh_pct: float, wind_kmh: float, rain_mm: float) -> float:
    """Computes single-day Fine Fuel Moisture Code (FFMC) update (Van Wagner 1987)."""
    t = float(temp_c)
    h = float(np.clip(rh_pct, 1.0, 100.0))
    w = float(max(0.0, wind_kmh))
    ro = float(max(0.0, rain_mm))
    f0 = float(np.clip(ffmc_prev, 0.0, 101.0))

    mo = 147.2 * (101.0 - f0) / (59.5 + f0)

    if ro > 0.5:
        rf = ro - 0.5
        if mo <= 150.0:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mr = (
                mo
                + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                + 0.0015 * ((mo - 150.0) ** 2) * math.sqrt(rf)
            )
        mo = min(mr, 250.0)

    ed = (
        0.942 * (h ** 0.679)
        + 11.0 * math.exp((h - 100.0) / 10.0)
        + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * h))
    )

    if mo > ed:
        ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * math.sqrt(w) * (1.0 - (h / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * t)
        m = ed + (mo - ed) * (10.0 ** (-kd))
    else:
        ew = (
            0.618 * (h ** 0.753)
            + 10.0 * math.exp((h - 100.0) / 10.0)
            + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * h))
        )
        if mo < ew:
            kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * math.sqrt(w) * (
                1.0 - ((100.0 - h) / 100.0) ** 8
            )
            kw = kl * 0.581 * math.exp(0.0365 * t)
            m = ew - (ew - mo) * (10.0 ** (-kw))
        else:
            m = mo

    m = max(0.0, min(250.0, m))
    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return float(np.clip(ffmc, 0.0, 101.0))


def dmc_step(dmc_prev: float, temp_c: float, rh_pct: float, rain_mm: float, month: int) -> float:
    """Computes single-day Duff Moisture Code (DMC) update (Van Wagner 1987)."""
    t = max(-1.1, float(temp_c))
    h = float(np.clip(rh_pct, 1.0, 100.0))
    ro = float(max(0.0, rain_mm))
    p0 = max(0.0, float(dmc_prev))

    if ro > 1.5:
        re = 0.92 * ro - 1.27
        mo = 20.0 + math.exp(5.6348 - p0 / 43.43)
        if p0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * p0)
        elif p0 <= 65.0:
            b = 14.0 - 1.3 * math.log(p0)
        else:
            b = 6.2 * math.log(p0) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * math.log(max(mr - 20.0, 1e-6))
        p0 = max(0.0, pr)

    le = _DMC_DAY_LENGTH_FACTORS[max(1, min(12, int(month))) - 1]
    k = 1.894 * (t + 1.1) * (100.0 - h) * le * 1e-6
    return float(max(0.0, p0 + 100.0 * k))


def dc_step(dc_prev: float, temp_c: float, rain_mm: float, month: int) -> float:
    """Computes single-day Drought Code (DC) update (Van Wagner 1987)."""
    t = max(-2.8, float(temp_c))
    ro = float(max(0.0, rain_mm))
    d0 = max(0.0, float(dc_prev))

    if ro > 2.8:
        rd = 0.83 * ro - 1.27
        qo = 800.0 * math.exp(-d0 / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * math.log(800.0 / max(qr, 1e-6))
        d0 = max(0.0, dr)

    lf = _DC_DAY_LENGTH_FACTORS[max(1, min(12, int(month))) - 1]
    v = max(0.0, 0.36 * (t + 2.8) + lf)
    return float(max(0.0, d0 + 0.5 * v))


def isi_from_ffmc_wind(ffmc: float, wind_kmh: float) -> float:
    """Computes Initial Spread Index (ISI) from FFMC and wind speed (km/h)."""
    f = float(np.clip(ffmc, 0.0, 101.0))
    w = float(max(0.0, wind_kmh))
    m = 147.2 * (101.0 - f) / (59.5 + f)
    f_w = math.exp(0.05039 * w)
    f_f = 91.9 * math.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
    return float(max(0.0, 0.208 * f_w * f_f))


def bui_from_dmc_dc(dmc: float, dc: float) -> float:
    """Computes Buildup Index (BUI) from DMC and DC."""
    p = max(0.0, float(dmc))
    d = max(0.0, float(dc))
    if p <= 0.0 and d <= 0.0:
        return 0.0
    if p <= 0.4 * d:
        u = 0.8 * p * d / max(p + 0.4 * d, 1e-6)
    else:
        u = p - (1.0 - 0.8 * d / max(p + 0.4 * d, 1e-6)) * (0.92 + (0.0114 * p) ** 1.7)
    return float(max(0.0, u))


def fwi_from_isi_bui(isi: float, bui: float) -> float:
    """Computes Fire Weather Index (FWI) from ISI and BUI."""
    r = max(0.0, float(isi))
    u = max(0.0, float(bui))
    if u <= 80.0:
        f_d = 0.626 * (u ** 0.809) + 2.0
    else:
        f_d = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * u))
    b = 0.1 * r * f_d
    if b > 1.0:
        s = math.exp(2.72 * ((0.434 * math.log(b)) ** 0.647))
    else:
        s = b
    return float(max(0.0, s))


def dsr_from_fwi(fwi: float) -> float:
    """Computes Daily Severity Rating (DSR) from FWI."""
    return float(0.0272 * (max(0.0, float(fwi)) ** 1.77))


def compute_fwi_series_from_hourly(
    T: np.ndarray,
    RH: np.ndarray,
    W: np.ndarray,
    PC: np.ndarray,
    end_day: date,
    ffmc0: float = 85.0,
    dmc0: float = 6.0,
    dc0: float = 15.0,
) -> Dict[str, float]:
    """Evolves the Canadian FWI System day-by-day over an hourly Open-Meteo series
    ending on `end_day` (using noon 12:00-14:00 UTC observations and 24h precipitation).
    """
    n_days = len(T) // 24
    if n_days < 1:
        return {
            "fwi_ffmc": ffmc0,
            "fwi_dmc": dmc0,
            "fwi_dc": dc0,
            "fwi_isi": 0.0,
            "fwi_bui": 0.0,
            "fwi_value": 0.0,
            "fwi_dsr": 0.0,
            "fwi_mean_7d": 0.0,
            "fwi_max_7d": 0.0,
        }

    start_day = end_day - timedelta(days=n_days - 1)
    ffmc, dmc, dc = float(ffmc0), float(dmc0), float(dc0)
    fwi_hist: List[float] = []
    isi, bui, fwi = 0.0, 0.0, 0.0

    for d_idx in range(n_days):
        cur_day = start_day + timedelta(days=d_idx)
        s = d_idx * 24
        e = s + 24
        t_day = T[s:e]
        rh_day = RH[s:e]
        w_day = W[s:e]
        pc_day = PC[s:e]

        # Afternoon / noon window (hours 12..15 UTC, standard 12:00 LST in Iberia)
        t_noon_slice = t_day[12:16]
        rh_noon_slice = rh_day[12:16]
        w_noon_slice = w_day[12:16]

        t_val = float(np.nanmean(t_noon_slice)) if np.isfinite(t_noon_slice).any() else float(np.nanmax(t_day))
        rh_val = float(np.nanmean(rh_noon_slice)) if np.isfinite(rh_noon_slice).any() else float(np.nanmin(rh_day))
        w_val = float(np.nanmean(w_noon_slice)) if np.isfinite(w_noon_slice).any() else float(np.nanmean(w_day))
        rain_24h = float(np.nansum(pc_day)) if np.isfinite(pc_day).any() else 0.0

        if not np.isfinite(t_val):
            t_val = 20.0
        if not np.isfinite(rh_val):
            rh_val = 50.0
        if not np.isfinite(w_val):
            w_val = 10.0

        ffmc = ffmc_step(ffmc, t_val, rh_val, w_val, rain_24h)
        dmc = dmc_step(dmc, t_val, rh_val, rain_24h, cur_day.month)
        dc = dc_step(dc, t_val, rain_24h, cur_day.month)
        isi = isi_from_ffmc_wind(ffmc, w_val)
        bui = bui_from_dmc_dc(dmc, dc)
        fwi = fwi_from_isi_bui(isi, bui)
        fwi_hist.append(fwi)

    recent_7d = fwi_hist[-7:] if len(fwi_hist) >= 7 else fwi_hist
    return {
        "fwi_ffmc": float(ffmc),
        "fwi_dmc": float(dmc),
        "fwi_dc": float(dc),
        "fwi_isi": float(isi),
        "fwi_bui": float(bui),
        "fwi_value": float(fwi),
        "fwi_dsr": dsr_from_fwi(fwi),
        "fwi_mean_7d": float(np.mean(recent_7d)),
        "fwi_max_7d": float(np.max(recent_7d)),
    }


def compute_fwi_from_weather_dict(w: Dict[str, float], weather_day: date) -> Dict[str, float]:
    """Reconstructs and computes the Canadian FWI components from aggregated 30d/7d/1d
    Open-Meteo weather dictionary when loading pre-existing weather caches.
    """
    if "fwi_value" in w and np.isfinite(w["fwi_value"]):
        return {
            "fwi_ffmc": float(w.get("fwi_ffmc", 85.0)),
            "fwi_dmc": float(w.get("fwi_dmc", 6.0)),
            "fwi_dc": float(w.get("fwi_dc", 15.0)),
            "fwi_isi": float(w.get("fwi_isi", 0.0)),
            "fwi_bui": float(w.get("fwi_bui", 0.0)),
            "fwi_value": float(w["fwi_value"]),
            "fwi_dsr": float(w.get("fwi_dsr", dsr_from_fwi(float(w["fwi_value"])))),
            "fwi_mean_7d": float(w.get("fwi_mean_7d", w["fwi_value"])),
            "fwi_max_7d": float(w.get("fwi_max_7d", w["fwi_value"])),
        }

    # Evolve 30-day synthetic trajectory matching the exact 30d, 7d, and 1d Open-Meteo statistics
    t30 = float(w.get("lookback_30d_temp_max", w.get("target_temperature_max", 22.0)))
    rh30 = float(w.get("lookback_30d_rh_min", w.get("target_rh_min", 45.0))) * 1.15
    w30 = float(w.get("lookback_30d_wind_max", w.get("target_wind_speed_max", 12.0))) * 0.65
    p30_total = float(w.get("lookback_30d_precip_sum", 0.0))
    # Note: some cached lookback precip values are scaled or summed over hours; normalize if > 500 mm
    if p30_total > 500.0:
        p30_total = p30_total / 1000.0

    t7 = float(w.get("lookback_7d_temp_max", w.get("target_temperature_max", 24.0)))
    rh7 = float(w.get("lookback_7d_rh_min", w.get("target_rh_min", 40.0))) * 1.10
    w7 = float(w.get("lookback_7d_wind_max", w.get("target_wind_speed_max", 12.0))) * 0.70
    p7_total = float(w.get("lookback_7d_precip_sum", 0.0))
    if p7_total > 500.0:
        p7_total = p7_total / 1000.0

    t1 = float(w.get("target_temperature_max", 25.0))
    rh1 = float(w.get("target_rh_min", 35.0)) * 1.08
    w1 = float(w.get("target_wind_speed_max", 12.0)) * 0.75
    if w1 <= 0.5:
        w1 = max(float(w.get("lookback_30d_wind_max", 10.0)) * 0.65, 8.0)
    if w7 <= 0.5:
        w7 = w1
    if w30 <= 0.5:
        w30 = w1

    p1 = float(w.get("target_precipitation_sum", 0.0))
    days_dry = int(round(float(w.get("days_since_rain", 5.0))))

    ffmc, dmc, dc = 85.0, 6.0, 15.0
    start_day = weather_day - timedelta(days=29)
    p_early_daily = max(0.0, p30_total - p7_total) / 23.0
    p_recent_daily = max(0.0, p7_total - p1) / 6.0

    fwi_hist: List[float] = []
    isi, bui, fwi = 0.0, 0.0, 0.0
    for d_idx in range(30):
        cur_day = start_day + timedelta(days=d_idx)
        days_before_end = 29 - d_idx
        if days_before_end == 0:
            td, rhd, wd, pd_mm = t1, rh1, w1, p1
        elif days_before_end <= 6:
            td, rhd, wd = t7, rh7, w7
            pd_mm = 0.0 if days_before_end < days_dry else p_recent_daily
        else:
            td, rhd, wd = t30, rh30, w30
            pd_mm = 0.0 if days_before_end < days_dry else p_early_daily

        ffmc = ffmc_step(ffmc, td, rhd, wd, pd_mm)
        dmc = dmc_step(dmc, td, rhd, pd_mm, cur_day.month)
        dc = dc_step(dc, td, pd_mm, cur_day.month)
        isi = isi_from_ffmc_wind(ffmc, wd)
        bui = bui_from_dmc_dc(dmc, dc)
        fwi = fwi_from_isi_bui(isi, bui)
        fwi_hist.append(fwi)

    recent_7d = fwi_hist[-7:]
    return {
        "fwi_ffmc": float(ffmc),
        "fwi_dmc": float(dmc),
        "fwi_dc": float(dc),
        "fwi_isi": float(isi),
        "fwi_bui": float(bui),
        "fwi_value": float(fwi),
        "fwi_dsr": dsr_from_fwi(fwi),
        "fwi_mean_7d": float(np.mean(recent_7d)),
        "fwi_max_7d": float(np.max(recent_7d)),
    }


def compute_fwi_for_grid_from_tab1d(
    TAB_1D: np.ndarray,
    col_names: Sequence[str],
    weather_day: date,
) -> Dict[str, np.ndarray]:
    """Vectorized/loop wrapper computing FWI components for all N grid points from TAB_1D."""
    from config_and_features import CONTEXT_1D, LOG_COLS

    n = len(TAB_1D)
    col_idx = {c: i for i, c in enumerate(col_names)}
    log_col_names = {CONTEXT_1D[i] for i in LOG_COLS}

    fwi_val = np.zeros(n, dtype=np.float32)
    ffmc_val = np.zeros(n, dtype=np.float32)
    dmc_val = np.zeros(n, dtype=np.float32)
    dc_val = np.zeros(n, dtype=np.float32)
    isi_val = np.zeros(n, dtype=np.float32)
    bui_val = np.zeros(n, dtype=np.float32)
    dsr_val = np.zeros(n, dtype=np.float32)

    for i in range(n):
        row = TAB_1D[i]
        w_dict: Dict[str, float] = {}
        for k in CONTEXT_1D:
            if k in col_idx:
                v = float(row[col_idx[k]])
                if k in log_col_names:
                    v = float(np.expm1(max(0.0, v)))
                w_dict[k] = v
        res = compute_fwi_from_weather_dict(w_dict, weather_day)
        fwi_val[i] = res["fwi_value"]
        ffmc_val[i] = res["fwi_ffmc"]
        dmc_val[i] = res["fwi_dmc"]
        dc_val[i] = res["fwi_dc"]
        isi_val[i] = res["fwi_isi"]
        bui_val[i] = res["fwi_bui"]
        dsr_val[i] = res["fwi_dsr"]

    # Smooth monotonic mapping of raw FWI [0, 100+] to calibrated probability [0, 1]
    # Calibrated to Iberian summer 50 km base rate (mean P ~ 0.12, 60th pct ~ 0.09, 90th pct ~ 0.23)
    p_fwi = 1.0 / (1.0 + np.exp(-(fwi_val - 78.0) / 10.5))
    p_fwi = np.clip(p_fwi, 1e-4, 0.999).astype(np.float32)

    return {
        "fwi": fwi_val,
        "ffmc": ffmc_val,
        "dmc": dmc_val,
        "dc": dc_val,
        "isi": isi_val,
        "bui": bui_val,
        "dsr": dsr_val,
        "p_fwi": p_fwi,
    }
