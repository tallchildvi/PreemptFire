"""Feature definitions and transformation routines for PreemptFire."""
from __future__ import annotations

import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import nbinom

BASE_DIR = Path(__file__).resolve().parent

CONFIG: Dict[str, Any] = {
    "FIRMS_MAP_KEY": os.getenv("FIRMS_MAP_KEY", ""),
    "GEE_PROJECT": os.getenv("GEE_PROJECT", ""),
    "GEE_SERVICE_ACCOUNT": os.getenv("GEE_SERVICE_ACCOUNT", ""),
    "GEE_KEY_FILE": os.getenv("GEE_KEY_FILE", ""),
    "FIRMS_HISTORY": os.getenv("PF_FIRMS_HISTORY", str(BASE_DIR / "firms_points_2017_2025.csv.gz")),
    "FIRMS_COVERED": os.getenv("PF_FIRMS_COVERED", str(BASE_DIR / "firms_covered_days.csv")),
    "HISTORICAL_FIRMS_CSV": os.getenv("PF_FIRMS_HISTORY", str(BASE_DIR / "firms_points_2017_2025.csv.gz")),
    "HISTORICAL_COVERED_CSV": os.getenv("PF_FIRMS_COVERED", str(BASE_DIR / "firms_covered_days.csv")),
    "POPULATION_CACHE_CSV": os.getenv("PF_POPULATION_CACHE", str(BASE_DIR / "population_cache.csv")),
    "POPULATION_CACHE": os.getenv("PF_POPULATION_CACHE", str(BASE_DIR / "population_cache.csv")),
    "WEATHER_GAP_DAYS": int(os.getenv("PF_WEATHER_GAP_DAYS", "10")),
    "HISTORY_DAYS": int(os.getenv("PF_HISTORY_DAYS", "30")),
    "CLIM_WINDOW_DAYS": int(os.getenv("PF_CLIM_WINDOW_DAYS", "15")),
    "MODEL_DIR": os.getenv("PF_MODEL_DIR", str(BASE_DIR / "models")),
    "MINI_DATASET_DIR": os.getenv("PF_MINI_DATASET_DIR", str(BASE_DIR / "mini_dataset")),
}

IBERIA_BBOX = "-9.5,35.9,3.4,43.9"  # lon_min,lat_min,lon_max,lat_max
R_EARTH = 6371.0088
RADII: Tuple[int, ...] = (5, 10, 25, 50)
RADII_KM: Tuple[int, ...] = RADII
PRIMARY_R = 50
HIST_DAYS = 30
HIST_WINDOWS: Tuple[int, ...] = (1, 3, 7, 14, 30)
CLIM_WINDOW = 15
INTERVAL_ALPHA = 0.1
LEVELS = ["Low", "Medium", "High"]
COLORS = ["#fee08b", "#f46d43", "#a50026"]

CHANNELS_2D: List[str] = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    "SAR_VV", "SAR_VH",
    "Elevation", "Slope", "Northness", "Eastness",
    "Travel_Time_Roads", "Travel_Time_Trails",
    "Dist_to_Railways", "Dist_to_Camps", "Dist_to_Powerlines",
    "Soil_Moisture",
]

S2_NAMES = [f"B{b}" for b in ("02", "03", "04", "05", "06", "07", "08", "8A", "11", "12")]
INDEX_NAMES = ["NDVI", "NBR", "NBR2", "NDWI", "EVI", "NDRE", "NDII", "MSI", "SAR_RATIO", "SAR_RVI"]
ALL_2D_NAMES = CHANNELS_2D + INDEX_NAMES
ALL_TORCH_CH = CHANNELS_2D + INDEX_NAMES
CI = {n: i for i, n in enumerate(ALL_2D_NAMES)}

CONTEXT_1D: List[str] = [
    "days_since_rain",
    "days_since_t0_pass",
    "int1_t0_to_target_precip_sum",
    "int1_t0_to_target_rh_max",
    "int1_t0_to_target_rh_mean",
    "int1_t0_to_target_rh_min",
    "int1_t0_to_target_temp_max",
    "int1_t0_to_target_temp_mean",
    "int1_t0_to_target_temp_min",
    "int1_t0_to_target_vpd_max",
    "int1_t0_to_target_vpd_mean",
    "int1_t0_to_target_wind_max",
    "int1_t0_to_target_wind_mean",
    "lookback_30d_precip_sum",
    "lookback_7d_precip_sum",
    "population_10km_log1p",
    "population_50km_log1p",
    "target_dew_point_mean",
    "target_precipitation_sum",
    "target_rh_mean",
    "target_rh_min",
    "target_surface_pressure_mean",
    "target_temperature_max",
    "target_temperature_mean",
    "target_temperature_min",
    "target_vpd_max",
    "target_vpd_mean",
    "target_wind_direction_mean",
    "target_wind_speed_max",
    "target_wind_speed_mean",
]

LOG_COLS = [i for i, c in enumerate(CONTEXT_1D) if "precip" in c or c.startswith("days_since")]
WIND_COL = CONTEXT_1D.index("target_wind_direction_mean")

TAB_1D_NAMES: List[str] = CONTEXT_1D + ["wind_dir_cos", "wind_dir_cos2", "doy_sin", "doy_cos"]
TAB_1D_COL_NAMES: List[str] = TAB_1D_NAMES

HIST_TAB_NAMES: List[str] = [f"h{w}d_{R}km" for w in HIST_WINDOWS for R in RADII] + ["hist_cov"]
HIST_COL_NAMES: List[str] = HIST_TAB_NAMES

CLIM_NAMES: List[str] = (
    [f"clim_lograte_{R}km" for R in RADII]
    + [f"clim_pday_{R}km" for R in RADII]
    + ["clim_ncov"]
)
CLIM_COL_NAMES: List[str] = CLIM_NAMES


def get_ft_names(H: int = 128) -> List[str]:
    r1, r2 = max(2, H // 24), max(6, H // 8)
    names = []
    for n in ALL_2D_NAMES:
        names.extend([f"{n}_mean", f"{n}_std", f"{n}_p10", f"{n}_p90", f"{n}_c{r1}", f"{n}_c{r2}"])
    for k in (0, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        names.append(f"scl_{k}")
    return names


FT_NAMES: List[str] = get_ft_names(128)
TAB_ALL_NAMES: List[str] = TAB_1D_NAMES + FT_NAMES


def to_xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def dist_km(xyz: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 2.0 * R_EARTH * np.arcsin(np.clip(np.linalg.norm(xyz - c, axis=1) / 2.0, 0.0, 1.0))


def circ(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 365 - d)


def raw_context_vector(ctx: Dict[str, float]) -> np.ndarray:
    return np.array([float(ctx.get(k, 0.0)) for k in CONTEXT_1D], dtype=np.float32)


def context_features(ctx: Dict[str, float] | np.ndarray, day: date) -> np.ndarray:
    if isinstance(ctx, dict):
        x = np.array([float(ctx[k]) for k in CONTEXT_1D], dtype=np.float64)
    else:
        x = np.asarray(ctx, dtype=np.float64).copy()
    raw_wd = x[WIND_COL]
    x[LOG_COLS] = np.log1p(np.clip(x[LOG_COLS], 0.0, None))
    w = np.deg2rad(raw_wd)
    doy = day.timetuple().tm_yday
    return np.r_[
        x,
        np.sin(w),
        np.cos(w),
        np.sin(2.0 * np.pi * doy / 365.25),
        np.cos(2.0 * np.pi * doy / 365.25),
    ].astype(np.float32)


def one_d_features(
    weather_dict: Dict[str, float] | np.ndarray,
    pop_10km_log1p: float,
    pop_50km_log1p: float,
    weather_day: date,
    s2_date: Optional[str | date] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(weather_dict, dict):
        ctx = dict(weather_dict)
    else:
        ctx = {k: float(weather_dict[i]) for i, k in enumerate(CONTEXT_1D[: len(weather_dict)])}
    ctx["population_10km_log1p"] = float(pop_10km_log1p)
    ctx["population_50km_log1p"] = float(pop_50km_log1p)
    if s2_date and str(s2_date) != "None":
        try:
            s2_d = datetime.strptime(str(s2_date)[:10], "%Y-%m-%d").date()
            ctx["days_since_t0_pass"] = float(max(1, min(60, (weather_day - s2_d).days)))
        except Exception:
            pass
    raw_1d = raw_context_vector(ctx)
    tab_1d = context_features(raw_1d, weather_day)
    return raw_1d, tab_1d


def transform_1d_for_nn(x1_raw: np.ndarray) -> np.ndarray:
    x1 = np.nan_to_num(np.asarray(x1_raw, dtype=np.float32).copy(), nan=0.0, posinf=0.0, neginf=0.0)
    if x1.ndim == 1:
        x1 = x1[None, :]
    x1[:, LOG_COLS] = np.log1p(np.clip(x1[:, LOG_COLS], 0.0, None))
    wd = np.deg2rad(x1[:, WIND_COL])
    x1[:, WIND_COL] = np.sin(wd)
    return np.column_stack([x1, np.cos(wd)]).astype(np.float32)


def history_many(
    pts_xyz: np.ndarray,
    fire: pd.DataFrame,
    covered: set,
    day: date,
    radii: Tuple[int, ...] = RADII,
    H: int = HIST_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(pts_xyz)
    HC = np.zeros((n, H, len(radii)), dtype=np.float32)
    obs = np.zeros(H, dtype=np.float32)
    ch_max = 2.0 * np.sin(max(radii) / (2.0 * R_EARTH))

    if len(fire) > 0:
        fd = pd.to_datetime(fire["acq_date"].astype(str).str[:10]).dt.date.values
        fxyz = to_xyz(fire["latitude"].values, fire["longitude"].values)
    else:
        fd = np.array([])
        fxyz = np.zeros((0, 3))

    for k in range(H):
        dd = day - timedelta(days=H - k)
        if dd not in covered:
            continue
        obs[k] = 1.0
        m = fd == dd
        if not m.any():
            continue
        tr = cKDTree(fxyz[m])
        for j, lst in enumerate(tr.query_ball_point(pts_xyz, r=ch_max)):
            if not lst:
                continue
            dk = dist_km(tr.data[lst], pts_xyz[j])
            for ri, R in enumerate(radii):
                HC[j, k, ri] = (dk <= R).sum()

    cols = [HC[:, H - w :, ri].sum(1) for w in HIST_WINDOWS if w <= H for ri in range(len(radii))]
    X_hist = np.column_stack(cols + [np.full(n, obs.mean(), dtype=np.float32)]).astype(np.float32)
    return X_hist, HC


def history_single(
    fire: pd.DataFrame,
    covered: set,
    lat: float,
    lon: float,
    day: date,
    radii: Tuple[int, ...] = RADII,
    H: int = HIST_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    xyz = to_xyz([lat], [lon])
    X_hist, HC = history_many(xyz, fire, covered, day, radii=radii, H=H)
    return X_hist[0], HC[0]


def build_seq_and_static(
    HC: np.ndarray,
    TAB_1D: np.ndarray,
    CLIM_FEATS: np.ndarray,
    H: int = HIST_DAYS,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(TAB_1D)
    obs = np.ones((n, H, 1), dtype=np.float32)
    seq = np.concatenate([np.log1p(np.maximum(HC, 0.0)) / 3.0, obs], axis=2).astype(np.float32)
    if TAB_1D.shape[1] >= 38:
        st_1d = TAB_1D[:, :38]
    else:
        st_1d = TAB_1D
    clim_cols = 47 - st_1d.shape[1]
    static_1d = np.hstack([st_1d, CLIM_FEATS[:, :clim_cols]]).astype(np.float32)
    return seq, static_1d


class ClimatologyIndex:
    def __init__(self, hist_path: str | Path, cov_path: Optional[str | Path] = None):
        self.hist_path = Path(hist_path)
        self.cov_path = Path(cov_path) if cov_path else None
        self._loaded = False
        self.hxyz = np.zeros((0, 3))
        self.p_year = np.zeros(0, dtype=np.int32)
        self.p_doy = np.zeros(0, dtype=np.int32)
        self.p_day = np.zeros(0, dtype=np.int64)
        self.c_year = np.zeros(0, dtype=np.int32)
        self.c_doy = np.zeros(0, dtype=np.int32)
        self.tree: Optional[cKDTree] = None

    def ensure_loaded(self) -> bool:
        if self._loaded:
            return self.tree is not None
        self._loaded = True
        if not self.hist_path.exists():
            alt_gz = self.hist_path.with_suffix(self.hist_path.suffix + ".gz")
            alt_csv = Path(str(self.hist_path).replace(".csv.gz", ".csv"))
            if alt_gz.exists():
                self.hist_path = alt_gz
            elif alt_csv.exists():
                self.hist_path = alt_csv
            else:
                return False

        hist = pd.read_csv(self.hist_path)
        pdt = pd.to_datetime(hist["acq_date"].astype(str).str[:10])
        self.p_year = pdt.dt.year.values
        self.p_doy = pdt.dt.dayofyear.values
        self.p_day = pdt.values.astype("datetime64[D]").astype(np.int64)
        self.hxyz = to_xyz(hist["latitude"].values, hist["longitude"].values)
        self.tree = cKDTree(self.hxyz)

        if self.cov_path and self.cov_path.exists():
            cov_df = pd.read_csv(self.cov_path)
            cdt = pd.to_datetime(np.unique(cov_df["acq_date"].astype(str).str[:10]))
        else:
            cdt = pd.date_range(pdt.min(), pdt.max())
        self.c_year = cdt.year.values
        self.c_doy = cdt.dayofyear.values
        return True

    def query_many(
        self,
        pts_xyz: np.ndarray,
        day: date,
        radii: Tuple[int, ...] = RADII,
        window: int = CLIM_WINDOW,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        n = len(pts_xyz)
        rate = np.zeros((n, len(radii)), dtype=np.float32)
        pday = np.zeros((n, len(radii)), dtype=np.float32)

        if not self.ensure_loaded():
            ncov = 248.0
            X_clim = np.column_stack([
                np.log(rate + 1e-3),
                pday,
                np.full(n, np.log1p(ncov), dtype=np.float32),
            ]).astype(np.float32)
            return X_clim, rate, pday, ncov

        year, doy = day.year, day.timetuple().tm_yday
        ok_c = (self.c_year != year) & (circ(self.c_doy, doy) <= window)
        ncov = float(ok_c.sum())

        ok_p = (self.p_year != year) & (circ(self.p_doy, doy) <= window)
        if ok_p.any() and ncov > 0:
            sub_xyz = self.hxyz[ok_p]
            sub_day = self.p_day[ok_p]
            sub_tree = cKDTree(sub_xyz)
            ch_max = 2.0 * np.sin(max(radii) / (2.0 * R_EARTH))
            for j, lst in enumerate(sub_tree.query_ball_point(pts_xyz, r=ch_max)):
                if not lst:
                    continue
                lst_arr = np.asarray(lst, dtype=int)
                dk = dist_km(sub_xyz[lst_arr], pts_xyz[j])
                for ri, R in enumerate(radii):
                    m = dk <= R
                    rate[j, ri] = m.sum() / ncov
                    pday[j, ri] = len(np.unique(sub_day[lst_arr][m])) / ncov

        X_clim = np.column_stack([
            np.log(rate + 1e-3),
            pday,
            np.full(n, np.log1p(ncov), dtype=np.float32),
        ]).astype(np.float32)
        return X_clim, rate, pday, ncov


_CLIM_INDEX: Optional[ClimatologyIndex] = None


def get_climatology_index() -> ClimatologyIndex:
    global _CLIM_INDEX
    if _CLIM_INDEX is None:
        _CLIM_INDEX = ClimatologyIndex(CONFIG["FIRMS_HISTORY"], CONFIG["FIRMS_COVERED"])
    return _CLIM_INDEX


def climatology_many(
    pts_xyz: np.ndarray,
    day: date,
    radii: Tuple[int, ...] = RADII,
    window: int = CLIM_WINDOW,
    hist_path: Optional[str | Path] = None,
    cov_path: Optional[str | Path] = None,
) -> np.ndarray:
    idx = get_climatology_index()
    if hist_path and Path(hist_path).exists():
        idx.hist_path = Path(hist_path)
    if cov_path and Path(cov_path).exists():
        idx.cov_path = Path(cov_path)
    X_clim, _, _, _ = idx.query_many(pts_xyz, day, radii=radii, window=window)
    return X_clim


def fit_nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    yv = np.round(np.clip(np.asarray(y, dtype=float), 0, None))
    mu_v = np.clip(np.asarray(mu, dtype=float), 1e-6, None)

    def nb_nll(alpha_val):
        r = 1.0 / max(alpha_val, 1e-4)
        return float(-np.mean(nbinom.logpmf(yv, r, r / (r + mu_v))))

    try:
        res = minimize_scalar(lambda la: nb_nll(math.exp(la)), bounds=(-6, 4), method="bounded")
        return float(math.exp(res.x))
    except Exception:
        return 4.0


fit_alpha = fit_nb_alpha


def detect_scales(sample_x2d: np.ndarray) -> Tuple[str, str]:
    s2 = sample_x2d[:, :10].astype(np.float32)
    s2 = s2[np.isfinite(s2)]
    sar = sample_x2d[:, 10:12].astype(np.float32)
    sar = sar[np.isfinite(sar)]
    s2_scale = "0_1" if (len(s2) == 0 or np.percentile(np.abs(s2), 99) <= 2.0) else "0_10000"
    sar_scale = "db" if (len(sar) > 0 and np.percentile(sar, 1) < -1e-3) else "linear"
    return s2_scale, sar_scale


def build_2d_np(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        x = x[None, ...]
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    s2_scale, sar_scale = detect_scales(x)
    b = {n: x[:, i].copy() for i, n in enumerate(CHANNELS_2D)}
    if s2_scale == "0_10000":
        for n in S2_NAMES:
            b[n] = b[n] / 10000.0
    vv, vh = b["SAR_VV"], b["SAR_VH"]
    if sar_scale == "db":
        vv, vh = 10.0 ** (vv / 10.0), 10.0 ** (vh / 10.0)

    r = lambda a, c, e=1e-6: a / (c + e)
    nir, red, blue, sw1, sw2, re_ = b["B08"], b["B04"], b["B02"], b["B11"], b["B12"], b["B05"]
    den = nir + 6.0 * red - 7.5 * blue + 1.0
    den = np.where(np.abs(den) < 1e-3, np.where(den >= 0, 1e-3, -1e-3), den)
    d = sw1 - sw2
    den2 = nir + d
    den2 = np.where(np.abs(den2) < 1e-3, np.where(den2 >= 0, 1e-3, -1e-3), den2)

    idx = [
        r(nir - red, nir + red),
        r(nir - sw1, nir + sw1),
        r(nir - sw2, nir + sw2),
        r(sw1 - sw2, sw1 + sw2),
        np.clip(2.5 * (nir - red) / den, -2.0, 2.0),
        r(nir - re_, nir + re_),
        np.clip((sw1 + 1e-3) / (nir + 1e-3), 0.0, 10.0),
        np.clip((nir - d) / den2, -2.0, 2.0),
        np.clip(r(vh, vv), 0.0, 5.0),
        np.clip(r(4.0 * vh, vv + vh), 0.0, 4.0),
    ]
    out = np.concatenate([x, np.stack(idx, axis=1)], axis=1)
    for n in ("SAR_VV", "SAR_VH"):
        a = out[:, CI[n]]
        out[:, CI[n]] = np.clip(a if sar_scale == "db" else 10.0 * np.log10(np.clip(a, 1e-3, None)), -30.0, 10.0)
    for n in ("Dist_to_Railways", "Dist_to_Camps", "Dist_to_Powerlines"):
        out[:, CI[n]] = np.log1p(np.clip(out[:, CI[n]], 0.0, None))
    return out.astype(np.float32)


def compute_2d_tab_features(x2d: np.ndarray, scl: np.ndarray) -> np.ndarray:
    if x2d.ndim == 3:
        x2d = x2d[None, ...]
    if scl.ndim == 2:
        scl = scl[None, ...]
    _, _, H, W = x2d.shape
    cy, cx = H // 2, W // 2
    r1, r2 = max(2, H // 24), max(6, H // 8)

    x = build_2d_np(x2d)
    f = {}
    for i, n in enumerate(ALL_2D_NAMES):
        a = x[:, i]
        f[f"{n}_mean"] = a.mean((1, 2))
        f[f"{n}_std"] = a.std((1, 2))
        q = np.percentile(a.reshape(len(a), -1), [10, 90], axis=1)
        f[f"{n}_p10"], f[f"{n}_p90"] = q[0], q[1]
        f[f"{n}_c{r1}"] = a[:, cy - r1 : cy + r1 + 1, cx - r1 : cx + r1 + 1].mean((1, 2))
        f[f"{n}_c{r2}"] = a[:, cy - r2 : cy + r2 + 1, cx - r2 : cx + r2 + 1].mean((1, 2))
    for k in (0, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        f[f"scl_{k}"] = (scl == k).mean((1, 2))
    names = list(f.keys())
    return np.column_stack([f[k] for k in names]).astype(np.float32)


def build_33ch_array(x2d: np.ndarray, scl: np.ndarray, size: int = 128) -> np.ndarray:
    if x2d.ndim == 3:
        x2d = x2d[None, ...]
    if scl.ndim == 2:
        scl = scl[None, ...]
    out32 = build_2d_np(x2d)
    water = (scl == 6).astype(np.float32)[:, None, :, :]
    arr33 = np.concatenate([out32, water], axis=1).astype(np.float32)
    if arr33.shape[2] != size or arr33.shape[3] != size:
        # Center crop or pad to requested size
        H, W = arr33.shape[2], arr33.shape[3]
        if H >= size and W >= size:
            sy = (H - size) // 2
            sx = (W - size) // 2
            arr33 = arr33[:, :, sy : sy + size, sx : sx + size]
    if arr33.ndim == 4 and arr33.shape[0] == 1:
        return arr33[0]
    return arr33
