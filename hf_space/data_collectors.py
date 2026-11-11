"""Live 1D & 2D data collectors for PreemptFire (Hugging Face Spaces & CLI).

Combines and optimizes the repository's collectors (`approach_b` branch):
  - WeatherFetcher (Open-Meteo ERA5 archive + forecast, with 50-point batching & disk cache)
  - PopulationFetcher (WorldPop 100m via GEE + `population_cache.csv` + KDTree fallback)
  - FIRMSFetcher (NASA FIRMS VIIRS_SNPP_SP / NRT with 5-day block disk cache)
  - Scene2DCollector (GridAligner + Sentinel-2 L2A + Sentinel-1 RTC SAR + Copernicus DEM 30m
    + Overpass OSM + Dijkstra Travel Time + ERA5-Land / Open-Meteo Soil Moisture)
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import heapq
from io import StringIO
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from urllib3.util import Retry

from config_and_features import (
    CHANNELS_2D,
    CONFIG,
    IBERIA_BBOX,
    R_EARTH,
    get_firms_keys,
    raw_context_vector,
    to_xyz,
)

GRID_SIZE_PX = 128
PIXEL_SCALE_METERS = 100.0
HALF_SCENE_METERS = (GRID_SIZE_PX // 2) * PIXEL_SCALE_METERS  # 6400.0 m
MAX_INVALID_RATIO = 0.20
MIN_DEM_VALID_RATIO = 0.98

_LAST_FIRMS_DIAGNOSTICS: Dict[str, Any] = {
    "keys_configured": 0,
    "active_key_prefix": "none",
    "last_src": None,
    "last_error": None,
    "last_status_code": None,
    "total_detections": 0,
}


import threading

_WEATHER_LOCK = threading.Lock()


def _make_session(retry_429: bool = True) -> requests.Session:
    s = requests.Session()
    codes = [429, 500, 502, 503, 504] if retry_429 else [500, 502, 503, 504]
    retries = Retry(total=3, backoff_factor=0.6, status_forcelist=codes)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def _open_meteo_urls_for_date(weather_day: date) -> List[str]:
    days_ago = (date.today() - weather_day).days
    if days_ago < 5:
        return [
            "https://api.open-meteo.com/v1/forecast",
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            "https://archive-api.open-meteo.com/v1/archive",
        ]
    if weather_day.year >= 2021:
        return [
            "https://archive-api.open-meteo.com/v1/archive",
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
        ]
    return ["https://archive-api.open-meteo.com/v1/archive"]


# 1. WEATHER COLLECTOR (Open-Meteo Single & Batched Multi-Point)

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


def _hourly_arr(h: dict, name: str) -> np.ndarray:
    v = h.get(name)
    if v is None:
        v = h.get(f"{name}_era5", [])
    return np.array([np.nan if x is None else x for x in v], dtype=np.float32)


def _agg_slice(
    temp: np.ndarray,
    rh: np.ndarray,
    wind: np.ndarray,
    precip: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    if len(temp) == 0 or np.isnan(temp).all():
        return {
            f"{prefix}_{k}": 0.0
            for k in ("temp_mean", "temp_max", "rh_min", "vpd_mean", "wind_max", "precip_sum")
        }
    svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    vpd = np.maximum(svp - svp * (rh / 100.0), 0.0)
    return {
        f"{prefix}_temp_mean": float(np.nanmean(temp)),
        f"{prefix}_temp_max": float(np.nanmax(temp)),
        f"{prefix}_rh_min": float(np.nanmin(rh)),
        f"{prefix}_vpd_mean": float(np.nanmean(vpd)),
        f"{prefix}_wind_max": float(np.nanmax(wind)),
        f"{prefix}_precip_sum": float(np.nansum(precip)),
    }


def _parse_open_meteo_payload(
    data: dict, weather_day: date, t0_date: date, rain_mm: float = 1.5
) -> Dict[str, float]:
    """Computes the exact 33 WeatherFetcher features from one Open-Meteo JSON object."""
    dp = np.array(
        [0.0 if p is None else p for p in data.get("daily", {}).get("precipitation_sum", [])],
        dtype=np.float32,
    )
    day_p = float(dp[-1]) if len(dp) else 0.0
    since = 0.0
    if day_p < rain_mm and len(dp) > 1:
        for p in reversed(dp[:-1]):
            if p >= rain_mm:
                break
            since += 1.0

    h = data.get("hourly", {})
    T, RH, DP, W, WD, PR, PC = (_hourly_arr(h, k) for k in HOURLY_VARS)
    if len(T) < 24 * 7:
        raise RuntimeError(f"Open-Meteo returned only {len(T)} valid hours")

    # Trim trailing all-NaN days if archive endpoint had a 1-day lag
    valid_idx = np.where(np.isfinite(T))[0]
    if len(valid_idx) == 0:
        raise RuntimeError("Open-Meteo hourly temperature is all NaN")
    last_valid_end = int(valid_idx[-1]) + 1
    if last_valid_end < len(T):
        T, RH, DP, W, WD, PR, PC = (
            arr[:last_valid_end] for arr in (T, RH, DP, W, WD, PR, PC)
        )

    t, rh, dp_, w, wd, pr = T[-24:], RH[-24:], DP[-24:], W[-24:], WD[-24:], PR[-24:]
    svp = 0.61078 * np.exp((17.27 * t) / (t + 237.3))
    vpd = np.maximum(svp - svp * (rh / 100.0), 0.0)
    v = wd[np.isfinite(wd)]
    wdir = (
        float((np.degrees(np.arctan2(np.mean(np.sin(np.radians(v))), np.mean(np.cos(np.radians(v))))) + 360.0) % 360.0)
        if len(v)
        else 0.0
    )

    out = {
        "target_temperature_max": float(np.nanmax(t)),
        "target_temperature_min": float(np.nanmin(t)),
        "target_temperature_mean": float(np.nanmean(t)),
        "target_dew_point_mean": float(np.nanmean(dp_)) if len(dp_) else 0.0,
        "target_rh_min": float(np.nanmin(rh)),
        "target_rh_mean": float(np.nanmean(rh)),
        "target_vpd_max": float(np.nanmax(vpd)),
        "target_vpd_mean": float(np.nanmean(vpd)),
        "target_wind_speed_max": float(np.nanmax(w)),
        "target_wind_speed_mean": float(np.nanmean(w)),
        "target_wind_direction_mean": wdir,
        "target_surface_pressure_mean": float(np.nanmean(pr)),
        "target_precipitation_sum": day_p,
        "days_since_rain": since,
    }
    gap = max(1, min(60, (weather_day - t0_date).days))
    out.update(_agg_slice(T[-gap * 24 :], RH[-gap * 24 :], W[-gap * 24 :], PC[-gap * 24 :], "int1_t0_to_target"))
    out["days_since_t0_pass"] = float(gap)
    out.update(_agg_slice(T[-168:], RH[-168:], W[-168:], PC[-168:], "lookback_7d"))
    out.update(_agg_slice(T[-720:], RH[-720:], W[-720:], PC[-720:], "lookback_30d"))
    return out


def fetch_weather(lat: float, lon: float, weather_day: date, t0_date: date, rain_mm: float = 1.5) -> Dict[str, float]:
    """Fetches Open-Meteo weather for a single point with instant multi-endpoint failover."""
    lookback_days = max(31, min(60, (weather_day - t0_date).days + 2))
    start = weather_day - timedelta(days=lookback_days)
    params = {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "start_date": start.isoformat(),
        "end_date": weather_day.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    sess = _make_session(retry_429=False)
    last_err = None
    for url in _open_meteo_urls_for_date(weather_day):
        try:
            r = sess.get(url, params=params, timeout=25)
            if r.status_code == 429:
                continue
            r.raise_for_status()
            return _parse_open_meteo_payload(r.json(), weather_day, t0_date, rain_mm=rain_mm)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Open-Meteo single-point weather fetch failed: {last_err}")


def _interpolate_weather_idw(
    pts: pd.DataFrame,
    cell_coords: List[Tuple[float, float]],
    cell_dicts: List[Dict[str, float]],
    k: int = 4,
) -> List[Dict[str, float]]:
    """Interpolates synoptic weather dictionaries from grid cells to target points via spherical KDTree IDW."""
    if len(cell_dicts) == 1:
        return [dict(cell_dicts[0]) for _ in range(len(pts))]

    c_lats = np.array([c[0] for c in cell_coords], dtype=np.float64)
    c_lons = np.array([c[1] for c in cell_coords], dtype=np.float64)
    tree = cKDTree(to_xyz(c_lats, c_lons))
    p_xyz = to_xyz(pts["lat"].values, pts["lon"].values)
    k_use = min(k, len(cell_dicts))
    dists, idxs = tree.query(p_xyz, k=k_use)
    if k_use == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    weights = 1.0 / np.maximum(dists, 1e-5) ** 2
    weights /= weights.sum(axis=1, keepdims=True)

    feat_keys = [fk for fk in cell_dicts[0].keys() if fk != "target_wind_direction_mean"]
    mat = np.array([[d.get(fk, 0.0) for fk in feat_keys] for d in cell_dicts], dtype=np.float32)
    wdirs = np.radians(np.array([d.get("target_wind_direction_mean", 0.0) for d in cell_dicts], dtype=np.float32))
    w_sin = np.sin(wdirs)
    w_cos = np.cos(wdirs)

    interp_mat = np.einsum("ij,ijk->ik", weights, mat[idxs])
    interp_sin = np.sum(weights * w_sin[idxs], axis=1)
    interp_cos = np.sum(weights * w_cos[idxs], axis=1)
    interp_wdir = (np.degrees(np.arctan2(interp_sin, interp_cos)) + 360.0) % 360.0

    out_list: List[Dict[str, float]] = []
    for i in range(len(pts)):
        row_d = {fk: float(interp_mat[i, j]) for j, fk in enumerate(feat_keys)}
        row_d["target_wind_direction_mean"] = float(interp_wdir[i])
        out_list.append(row_d)
    return out_list


def fetch_weather_cells(
    pts: pd.DataFrame,
    day: date,
    gap: int = 3,
    cell: float = 0.5,
    batch_size: int = 50,
    use_cache: bool = True,
    progress_cb=None,
) -> Tuple[List[Dict[str, float]], int]:
    """Batched Open-Meteo weather fetcher over synoptic cells with multi-endpoint failover,
    disk caching, and spherical KDTree IDW interpolation.
    """
    wd = day - timedelta(days=1)
    t0 = wd - timedelta(days=gap)
    key = (
        (np.round(pts["lat"] / cell) * cell).round(3).astype(str)
        + "_"
        + (np.round(pts["lon"] / cell) * cell).round(3).astype(str)
    )
    unique_keys = list(dict.fromkeys(key))
    cells = {k: (float(k.split("_")[0]), float(k.split("_")[1])) for k in unique_keys}

    cache_dirs = [
        Path(CONFIG["MINI_DATASET_DIR"]) / "weather_cache",
        Path(__file__).resolve().parent / "weather_cache",
        Path(__file__).resolve().parent / "data" / "weather_cache",
    ]
    cache_dirs[0].mkdir(parents=True, exist_ok=True)
    cache_file = cache_dirs[0] / f"weather_cells_{wd.isoformat()}_gap{gap}_c{cell}.json"

    with _WEATHER_LOCK:
        res: Dict[str, Optional[Dict[str, float]]] = {}
        if use_cache:
            for cdir in cache_dirs:
                for cand_cell in (cell, 0.25, 0.5):
                    cf = cdir / f"weather_cells_{wd.isoformat()}_gap{gap}_c{cand_cell}.json"
                    if cf.exists():
                        try:
                            cached = json.loads(cf.read_text(encoding="utf-8"))
                            if cand_cell == cell:
                                for k in unique_keys:
                                    if k in cached and cached[k] is not None:
                                        res[k] = cached[k]
                            elif len(cached) >= 50:
                                c_coords = [(float(ck.split("_")[0]), float(ck.split("_")[1])) for ck, cv in cached.items() if cv is not None]
                                c_dicts = [cv for cv in cached.values() if cv is not None]
                                if progress_cb:
                                    progress_cb(0.35, f"[Step 1/5] Open-Meteo ERA5 weather loaded from cache ({len(c_dicts)} cells).")
                                return _interpolate_weather_idw(pts, c_coords, c_dicts), len(c_dicts)
                        except Exception:
                            pass

        missing_keys = [k for k in unique_keys if k not in res]
        if missing_keys:
            lookback_days = max(31, min(60, gap + 2))
            start = wd - timedelta(days=lookback_days)
            candidate_urls = _open_meteo_urls_for_date(wd)
            throttled_urls: set = set()
            batches = [missing_keys[s : s + batch_size] for s in range(0, len(missing_keys), batch_size)]
            n_batches = len(batches)
            sess = _make_session(retry_429=False)

            if progress_cb:
                progress_cb(
                    0.05,
                    f"[Step 1/5] Open-Meteo ERA5 weather: fetching {n_batches} batches ({len(missing_keys)} synoptic cells)...",
                )

            done_cells = len(unique_keys) - len(missing_keys)
            for b_idx, b_keys in enumerate(batches):
                lats = ",".join(str(round(cells[k][0], 4)) for k in b_keys)
                lons = ",".join(str(round(cells[k][1], 4)) for k in b_keys)
                params = {
                    "latitude": lats,
                    "longitude": lons,
                    "start_date": start.isoformat(),
                    "end_date": wd.isoformat(),
                    "hourly": ",".join(HOURLY_VARS),
                    "daily": "precipitation_sum",
                    "timezone": "UTC",
                }
                urls_to_try = [u for u in candidate_urls if u not in throttled_urls] or list(candidate_urls)
                for url in urls_to_try:
                    try:
                        r = sess.get(url, params=params, timeout=30)
                        if r.status_code == 429:
                            throttled_urls.add(url)
                            time.sleep(1.2)
                            continue
                        r.raise_for_status()
                        payload = r.json()
                        items = payload if isinstance(payload, list) else [payload]
                        if len(items) == len(b_keys):
                            for k, item in zip(b_keys, items):
                                try:
                                    res[k] = _parse_open_meteo_payload(item, wd, t0)
                                except Exception:
                                    res[k] = None
                            break
                    except Exception:
                        continue

                # If candidate endpoints hit 429 on this batch, wait briefly and reset throttled list
                if len(throttled_urls) >= len(candidate_urls):
                    time.sleep(2.0)
                    throttled_urls.clear()

                done_batches = b_idx + 1
                done_cells += len(b_keys)
                frac = 0.05 + 0.30 * (done_batches / max(n_batches, 1))
                msg = (
                    f"[Step 1/5] Open-Meteo ERA5 weather: batch {done_batches}/{n_batches} complete "
                    f"({done_cells}/{len(unique_keys)} synoptic cells fetched)"
                )
                print(f"  {msg}")
                if progress_cb:
                    progress_cb(frac, msg)

            if use_cache:
                try:
                    valid_to_save = {k: v for k, v in res.items() if v is not None}
                    if valid_to_save:
                        cache_file.write_text(json.dumps(valid_to_save), encoding="utf-8")
                except Exception:
                    pass
        else:
            if progress_cb:
                progress_cb(0.35, f"[Step 1/5] Open-Meteo ERA5 weather loaded from cache ({len(unique_keys)} cells).")

    ok_keys = [k for k in unique_keys if res.get(k) is not None]
    if not ok_keys:
        raise RuntimeError("Open-Meteo weather fetch failed across all endpoints; check network or try again in 60s.")

    c_coords = [cells[k] for k in ok_keys]
    c_dicts = [res[k] for k in ok_keys]  # type: ignore[misc]
    return _interpolate_weather_idw(pts, c_coords, c_dicts), len(ok_keys)


# 2. LAST SENTINEL-2 PASS (STAC metadata only, for 1D single-point scoring)

def last_sentinel_pass(
    lat: float, lon: float, weather_day: date, lookback: int = 30, max_cloud: float = 60.0
) -> date:
    try:
        import planetary_computer
        import pystac_client

        cat = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        d = 0.0645
        items = list(
            cat.search(
                collections=["sentinel-2-l2a"],
                bbox=[lon - d, lat - d, lon + d, lat + d],
                datetime=f"{(weather_day - timedelta(days=lookback)).isoformat()}/{weather_day.isoformat()}",
                query={"eo:cloud_cover": {"lt": max_cloud}},
            ).items()
        )
        if items:
            return max(i.datetime.date() for i in items)
    except Exception as e:
        print(f"  [Sentinel STAC] metadata search failed ({e})")
    return weather_day - timedelta(days=3)


# 3. POPULATION COLLECTOR (Cache + GEE + KDTree Interpolation Fallback)

def _init_gee() -> bool:
    try:
        import ee

        key = CONFIG["GEE_KEY_FILE"]
        acct = CONFIG["GEE_SERVICE_ACCOUNT"]
        proj = CONFIG["GEE_PROJECT"] or None
        if key and acct:
            cred = (
                ee.ServiceAccountCredentials(acct, key_data=key)
                if key.lstrip().startswith("{")
                else ee.ServiceAccountCredentials(acct, key)
            )
            ee.Initialize(cred, project=proj)
        else:
            ee.Initialize(project=proj)
        return True
    except Exception as e:
        print(f"  [GEE] Earth Engine not initialized ({e})")
        return False


def fetch_population_single(lat: float, lon: float, year: int = 2020, cache_path: Optional[str] = None) -> Tuple[float, float]:
    """Fetches 10km and 50km log1p population for one point (checks cache first, then GEE, then KDTree fallback)."""
    cpath = Path(cache_path or CONFIG["POPULATION_CACHE"])
    if cpath.exists():
        try:
            cdf = pd.read_csv(cpath)
            if {"lat", "lon", "population_10km_log1p", "population_50km_log1p"} <= set(cdf.columns) and len(cdf):
                d2 = (cdf["lat"].values - lat) ** 2 + (cdf["lon"].values - lon) ** 2
                idx = int(np.argmin(d2))
                if d2[idx] < 1e-4:
                    return (
                        float(cdf["population_10km_log1p"].iloc[idx]),
                        float(cdf["population_50km_log1p"].iloc[idx]),
                    )
        except Exception:
            pass

    if _init_gee():
        try:
            import ee

            year_use = min(int(year), 2020)
            img = None
            try:
                f = ee.ImageCollection("projects/sat-io/open-datasets/WORLDPOP/pop").filter(
                    ee.Filter.calendarRange(year_use, year_use, "year")
                )
                if f.size().getInfo() > 0:
                    img = f.select(["population"]).mosaic()
            except Exception:
                pass
            if img is None:
                col = ee.ImageCollection("WorldPop/GP/100m/pop")
                f = col.filter(ee.Filter.calendarRange(year_use, year_use, "year"))
                img = (
                    f.select(["population"]).mosaic()
                    if f.size().getInfo() > 0
                    else col.select(["population"]).sort("system:time_start", False).first()
                )
            pt = ee.Geometry.Point([lon, lat])
            fc = ee.FeatureCollection([ee.Feature(pt.buffer(10000), {"id": "l"}), ee.Feature(pt.buffer(50000), {"id": "r"})])
            res = img.unmask(0).reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100).getInfo()
            p = {f["properties"]["id"]: max(0.0, float(f["properties"].get("sum", 0.0))) for f in res["features"]}
            return (
                float(np.log1p(p.get("l", 0.0))),
                float(np.log1p(p.get("r", 0.0))),
            )
        except Exception as e:
            print(f"  [Population GEE] single fetch failed ({e})")

    if cpath.exists():
        try:
            cdf = pd.read_csv(cpath).dropna(subset=["population_10km_log1p", "population_50km_log1p"])
            if len(cdf):
                d2 = (cdf["lat"].values - lat) ** 2 + ((cdf["lon"].values - lon) * np.cos(np.radians(lat))) ** 2
                idx = int(np.argmin(d2))
                return (
                    float(cdf["population_10km_log1p"].iloc[idx]),
                    float(cdf["population_50km_log1p"].iloc[idx]),
                )
        except Exception:
            pass

    return 8.85, 12.15


def fetch_population_points(
    pts: pd.DataFrame,
    cache: Optional[str] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    progress_cb=None,
) -> pd.DataFrame:
    """Returns DataFrame with columns [population_10km_log1p, population_50km_log1p] for all grid points."""
    if progress_cb:
        progress_cb(0.36, f"[Step 2/5] Population density (10km / 50km): checking cache for {len(pts)} locations...")
    cpath = Path(cache or CONFIG["POPULATION_CACHE"])
    if cpath.exists():
        try:
            c = pd.read_csv(cpath)
            c_round = c.copy()
            c_round["lat_r"] = c_round["lat"].round(4)
            c_round["lon_r"] = c_round["lon"].round(4)
            c_round = c_round.drop_duplicates(subset=["lat_r", "lon_r"])

            pts_r = pts.copy()
            pts_r["lat_r"] = pts_r["lat"].round(4)
            pts_r["lon_r"] = pts_r["lon"].round(4)
            m = pts_r.merge(
                c_round[["lat_r", "lon_r", "population_10km_log1p", "population_50km_log1p"]],
                on=["lat_r", "lon_r"],
                how="left",
            )
            if not m["population_10km_log1p"].isna().any():
                if progress_cb:
                    progress_cb(0.42, f"[Step 2/5] Population loaded from cache ({len(pts)} locations).")
                return m[["population_10km_log1p", "population_50km_log1p"]].reset_index(drop=True)
            valid_c = c.dropna(subset=["population_10km_log1p", "population_50km_log1p"]).reset_index(drop=True)
            if len(valid_c) > 0:
                tree = cKDTree(to_xyz(valid_c["lat"].values, valid_c["lon"].values))
                miss = m["population_10km_log1p"].isna().values
                _, idx = tree.query(to_xyz(pts["lat"].values[miss], pts["lon"].values[miss]), k=1)
                m.loc[miss, "population_10km_log1p"] = valid_c["population_10km_log1p"].values[idx]
                m.loc[miss, "population_50km_log1p"] = valid_c["population_50km_log1p"].values[idx]
                if progress_cb:
                    progress_cb(0.42, f"[Step 2/5] Population loaded and interpolated from cache ({len(pts)} locations).")
                return m[["population_10km_log1p", "population_50km_log1p"]].reset_index(drop=True)
        except Exception as e:
            print(f"  [Population Cache] warning reading {cpath}: {e}")

    if metadata_df is not None and {"population_10km_log1p", "population_50km_log1p"} <= set(metadata_df.columns):
        la_c = next(col for col in ("lat", "latitude") if col in metadata_df.columns)
        lo_c = next(col for col in ("lon", "longitude") if col in metadata_df.columns)
        m_sub = metadata_df[[la_c, lo_c, "population_10km_log1p", "population_50km_log1p"]].copy()
        m_sub.columns = ["lat", "lon", "population_10km_log1p", "population_50km_log1p"]
        m_sub["lat"] = m_sub["lat"].round(5)
        m_sub["lon"] = m_sub["lon"].round(5)
        m_sub = m_sub.drop_duplicates(subset=["lat", "lon"])
        merged = pts.merge(m_sub, on=["lat", "lon"], how="left")
        if not merged["population_10km_log1p"].isna().any():
            return merged[["population_10km_log1p", "population_50km_log1p"]].reset_index(drop=True)

    if _init_gee():
        try:
            import ee

            col = ee.ImageCollection("WorldPop/GP/100m/pop")
            f = col.filter(ee.Filter.calendarRange(2020, 2020, "year"))
            img = f.select(["population"]).mosaic().unmask(0)
            rows = []
            for s in range(0, len(pts), 200):
                ch = pts.iloc[s : s + 200]
                feats = [
                    ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(R), {"i": int(i), "R": R})
                    for i, r in ch.iterrows()
                    for R in (10000, 50000)
                ]
                out = (
                    img.reduceRegions(collection=ee.FeatureCollection(feats), reducer=ee.Reducer.sum(), scale=100)
                    .getInfo()["features"]
                )
                rows += [
                    (f_["properties"]["i"], f_["properties"]["R"], float(f_["properties"].get("sum", 0.0)))
                    for f_ in out
                ]
                msg = f"[Step 2/5] Population GEE WorldPop: {min(s + 200, len(pts))}/{len(pts)} points fetched"
                print(f"  {msg}")
                if progress_cb:
                    progress_cb(0.36 + 0.08 * (min(s + 200, len(pts)) / max(len(pts), 1)), msg)
            d = pd.DataFrame(rows, columns=["i", "R", "v"]).pivot(index="i", columns="R", values="v").clip(lower=0)
            res = pd.DataFrame(
                {
                    "population_10km_log1p": np.log1p(d[10000]).astype(np.float32),
                    "population_50km_log1p": np.log1p(d[50000]).astype(np.float32),
                }
            ).reindex(pts.index)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([pts.reset_index(drop=True), res.reset_index(drop=True)], axis=1).to_csv(cpath, index=False)
            return res
        except Exception as e:
            print(f"  [Population GEE] batch fetch failed ({e}), using fallback.")

    if progress_cb:
        progress_cb(0.44, "[Step 2/5] Population: using regional baseline priors (no GEE/cache found).")
    return pd.DataFrame(
        {
            "population_10km_log1p": np.full(len(pts), 8.85, dtype=np.float32),
            "population_50km_log1p": np.full(len(pts), 12.15, dtype=np.float32),
        },
        index=pts.index,
    )


def fetch_firms_days(
    first: date | float,
    last: date | float,
    third: Optional[date] = None,
    fourth: Optional[date] = None,
    radius_deg_lat: float = 0.6,
    bbox: Optional[str] = None,
    firms_keys: Optional[str | Sequence[str]] = None,
    progress_cb=None,
) -> Tuple[pd.DataFrame, set]:
    """Fetches daily NASA FIRMS detections in [first_date, last_date] with multi-key pool
    auto-failover, multi-source fallback (VIIRS_SNPP_NRT, NOAA20, NOAA21, SP), and disk caching.
    """
    if isinstance(first, (int, float)) and isinstance(last, (int, float)) and isinstance(third, date) and isinstance(fourth, date):
        lat, lon = float(first), float(last)
        first_dt, last_dt = third, fourth
    else:
        lat, lon = 0.0, 0.0
        first_dt, last_dt = first, last  # type: ignore[assignment]
        if bbox is None:
            bbox = IBERIA_BBOX

    first = first_dt
    last = last_dt
    frames = []
    covered: set = set()

    if progress_cb:
        progress_cb(0.46, f"[Step 3/5] NASA FIRMS fire history: checking local archive ({first} to {last})...")

    hist_path = Path(CONFIG["FIRMS_HISTORY"])
    cov_path = Path(CONFIG["FIRMS_COVERED"])
    if not hist_path.exists():
        alt = Path(str(hist_path).replace(".csv.gz", ".csv"))
        if alt.exists():
            hist_path = alt

    if hist_path.exists():
        try:
            hdf = pd.read_csv(hist_path)
            hdf["acq_date"] = hdf["acq_date"].astype(str).str[:10]
            h_min, h_max = hdf["acq_date"].min(), hdf["acq_date"].max()
            f_str, l_str = first.isoformat(), last.isoformat()
            if f_str <= h_max and l_str >= h_min:
                sub = hdf[(hdf["acq_date"] >= f_str) & (hdf["acq_date"] <= l_str)].copy()
                if bbox is None and (lat != 0 or lon != 0):
                    dlon = radius_deg_lat / np.cos(np.radians(lat))
                    sub = sub[
                        (sub["latitude"] >= lat - radius_deg_lat)
                        & (sub["latitude"] <= lat + radius_deg_lat)
                        & (sub["longitude"] >= lon - dlon)
                        & (sub["longitude"] <= lon + dlon)
                    ]
                if len(sub):
                    frames.append(sub[["latitude", "longitude", "acq_date"]])
                if cov_path.exists():
                    cdf = pd.read_csv(cov_path)
                    c_dates = set(pd.to_datetime(cdf["acq_date"].astype(str).str[:10]).dt.date)
                    d_iter = first
                    while d_iter <= last:
                        if d_iter in c_dates:
                            covered.add(d_iter)
                        d_iter += timedelta(days=1)
                else:
                    h_max_dt = datetime.strptime(h_max, "%Y-%m-%d").date()
                    h_min_dt = datetime.strptime(h_min, "%Y-%m-%d").date()
                    d_iter = max(first, h_min_dt)
                    while d_iter <= min(last, h_max_dt):
                        covered.add(d_iter)
                        d_iter += timedelta(days=1)
        except Exception as e:
            print(f"  [FIRMS Archive] warning reading local archive: {e}")

    all_req_days = {first + timedelta(days=i) for i in range((last - first).days + 1)}
    missing_days = sorted(all_req_days - covered)
    if not missing_days:
        pts = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
        _LAST_FIRMS_DIAGNOSTICS["total_detections"] = len(pts)
        if progress_cb:
            progress_cb(0.58, f"[Step 3/5] NASA FIRMS: loaded {len(pts)} detections across {len(covered)} days from archive.")
        return pts, covered

    keys = get_firms_keys(firms_keys)
    _LAST_FIRMS_DIAGNOSTICS["keys_configured"] = len(keys)
    if not keys:
        msg = "WARNING: No FIRMS_MAP_KEY configured. Please provide one or more NASA FIRMS MAP_KEYs in UI or environment."
        _LAST_FIRMS_DIAGNOSTICS["last_error"] = msg
        print(f"  [FIRMS API] {msg}")
        pts = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
        return pts, covered

    use_bbox = bbox
    if use_bbox is None:
        dlon = radius_deg_lat / np.cos(np.radians(lat))
        use_bbox = f"{lon - dlon:.3f},{lat - radius_deg_lat:.3f},{lon + dlon:.3f},{lat + radius_deg_lat:.3f}"

    firms_cache_dir = Path(CONFIG["MINI_DATASET_DIR"]) / "firms_cache"
    firms_cache_dir.mkdir(parents=True, exist_ok=True)
    bbox_tag = use_bbox.replace(",", "_").replace("-", "m").replace(".", "p")

    d = missing_days[0]
    last_miss = missing_days[-1]
    total_span = max(1, (last_miss - d).days + 1)
    sess = _make_session()

    while d <= last_miss:
        n = min(5, (last_miss - d).days + 1)
        days_ago = (date.today() - d).days
        if days_ago > 90:
            candidate_sources = ["VIIRS_SNPP_NRT", "VIIRS_SNPP_SP", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
        else:
            candidate_sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_SP", "MODIS_NRT"]

        done_span = (d - missing_days[0]).days
        if progress_cb:
            progress_cb(
                0.46 + 0.12 * (done_span / total_span),
                f"[Step 3/5] NASA FIRMS API: fetching {d.isoformat()} (+{n} days)...",
            )

        df = None
        # 1. Check local cache across candidate sources
        for cand_src in candidate_sources:
            cache_file = firms_cache_dir / f"firms_{cand_src}_{bbox_tag}_{d.isoformat()}_{n}d.csv"
            if cache_file.exists():
                try:
                    cand_df = pd.read_csv(cache_file)
                    df = cand_df
                    covered.update(d + timedelta(days=i) for i in range(n))
                    break
                except Exception:
                    pass

        # 2. Live fetch with automatic multi-key rotation and multi-source fallback
        if df is None:
            fetch_success = False
            last_err_detail = None

            for key_idx, key in enumerate(keys):
                key_masked = f"{key[:5]}...{key[-3:]}" if len(key) > 8 else "***"
                _LAST_FIRMS_DIAGNOSTICS["active_key_prefix"] = key_masked
                key_has_auth_error = False

                for src in candidate_sources:
                    cache_file = firms_cache_dir / f"firms_{src}_{bbox_tag}_{d.isoformat()}_{n}d.csv"
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{use_bbox}/{n}/{d.isoformat()}"
                    try:
                        r = sess.get(url, timeout=45)
                        resp_text = r.text.strip()
                        _LAST_FIRMS_DIAGNOSTICS["last_status_code"] = r.status_code

                        # Valid CSV returned (either with fire records or header-only for 0 fires)
                        if r.status_code == 200 and resp_text.startswith("latitude"):
                            df = pd.read_csv(StringIO(r.text))
                            df.to_csv(cache_file, index=False)
                            covered.update(d + timedelta(days=i) for i in range(n))
                            fetch_success = True
                            _LAST_FIRMS_DIAGNOSTICS["last_src"] = src
                            _LAST_FIRMS_DIAGNOSTICS["last_error"] = None
                            print(f"  [FIRMS API] Successfully fetched {len(df)} {src} detections for {d} (+{n}d) using Key #{key_idx + 1} ({key_masked})")
                            break

                        # Check for key exhaustion, transaction limit, or auth errors
                        is_key_err = (
                            r.status_code in (401, 403, 429)
                            or "limit reached" in resp_text.lower()
                            or "invalid map_key" in resp_text.lower()
                            or "unauthorized" in resp_text.lower()
                            or "exceeded" in resp_text.lower()
                        )
                        if is_key_err:
                            last_err_detail = f"Key #{key_idx + 1} ({key_masked}) error: HTTP {r.status_code} ({resp_text[:120]})"
                            print(f"  [FIRMS API] {last_err_detail}. Switching to next key...")
                            key_has_auth_error = True
                            break

                        # Otherwise source not available for this date/range, try next source
                        last_err_detail = f"{src} on {d} failed: HTTP {r.status_code} ({resp_text[:120]})"
                    except Exception as exc:
                        last_err_detail = f"{src} on {d} error: {exc}"
                        continue

                if fetch_success or not key_has_auth_error:
                    if fetch_success:
                        break

            if not fetch_success:
                _LAST_FIRMS_DIAGNOSTICS["last_error"] = last_err_detail or "All FIRMS keys and sources failed."
                print(f"  [FIRMS API] WARNING: Failed to fetch {d} (+{n}d) across {len(keys)} keys: {_LAST_FIRMS_DIAGNOSTICS['last_error']}")

        if df is not None and len(df) and "latitude" in df.columns:
            df["acq_date"] = df["acq_date"].astype(str).str[:10]
            if "confidence" in df.columns:
                df = df[~df["confidence"].astype(str).str.lower().isin(["l", "low"])]
            if "type" in df.columns:
                df = df[df["type"] == 0]
            frames.append(df[["latitude", "longitude", "acq_date"]])

        d += timedelta(days=n)

    pts = (
        pd.concat(frames, ignore_index=True).drop_duplicates()
        if frames
        else pd.DataFrame(columns=["latitude", "longitude", "acq_date"])
    )
    _LAST_FIRMS_DIAGNOSTICS["total_detections"] = len(pts)
    return pts, covered


# 5. FULL 2D SCENE COLLECTOR (For Image Models: tab_ssl_full, tab_full)

NEIGHBOR_OFFSETS = np.array(
    [
        [-1, 0, 1.0],
        [1, 0, 1.0],
        [0, -1, 1.0],
        [0, 1, 1.0],
        [-1, -1, 1.41421356],
        [-1, 1, 1.41421356],
        [1, -1, 1.41421356],
        [1, 1, 1.41421356],
    ],
    dtype=np.float64,
)


def _dijkstra_travel_time_50m(
    elevation: np.ndarray,
    sources: np.ndarray,
    passable_mask: np.ndarray,
    resolution: float = 50.0,
    max_time_cap_hours: float = 12.0,
) -> np.ndarray:
    """Exact Tobler hiking travel-time solver from src/data_pipeline/accessibility/solver.py."""
    rows, cols = elevation.shape
    time_grid = np.full((rows, cols), np.inf, dtype=np.float32)
    visited = np.zeros((rows, cols), dtype=bool)
    pq = []

    source_r, source_c = np.where((sources == 1) & (passable_mask == 1))
    if len(source_r) == 0:
        return np.full((rows, cols), max_time_cap_hours, dtype=np.float32)

    for r, c in zip(source_r, source_c):
        time_grid[r, c] = 0.0
        heapq.heappush(pq, (0.0, int(r), int(c)))

    while pq:
        curr_time, r, c = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True
        if curr_time >= max_time_cap_hours:
            continue
        z_curr = float(elevation[r, c])

        for i in range(8):
            nr = r + int(NEIGHBOR_OFFSETS[i, 0])
            nc = c + int(NEIGHBOR_OFFSETS[i, 1])
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and passable_mask[nr, nc] == 1:
                dx = resolution * NEIGHBOR_OFFSETS[i, 2]
                dh = float(elevation[nr, nc]) - z_curr
                slope = dh / dx
                if slope > 0.7813 or slope < -0.7002:
                    continue
                speed_kmh = 6.0 * math.exp(-3.5 * abs(slope + 0.05))
                if speed_kmh <= 0.01:
                    continue
                t_step = (dx / 1000.0) / speed_kmh
                if dh > 0.0:
                    t_step += dh / 600.0
                if sources[nr, nc] == 0:
                    t_step *= 2.5
                cand_time = curr_time + t_step
                if cand_time < time_grid[nr, nc]:
                    time_grid[nr, nc] = cand_time
                    heapq.heappush(pq, (cand_time, nr, nc))

    time_grid = np.where(np.isinf(time_grid), max_time_cap_hours, time_grid)
    return np.clip(time_grid, 0.0, max_time_cap_hours).astype(np.float32)


class Scene2DCollector:
    """Self-contained 22-channel 128x128 @ 100m scene collector matching `SingleSceneCollector` (`approach_b`).

    Features:
      - Caches static 9 channels (Elevation, Slope, Northness, Eastness, Travel_Time_Roads,
        Travel_Time_Trails, Dist_to_Railways, Dist_to_Camps, Dist_to_Powerlines) per (lat, lon)
        in `mini_dataset/static_2d_cache/` so repeated dates at the same location skip DEM + OSM!
      - Uses Open-Meteo `soil_moisture_0_to_7cm` automatically if GEE ERA5-Land is unavailable.
      - Uses Overpass API mirrors with local caching for OSM vectors.
    """

    SCL_SNOW = [11]
    SCL_INVALID = [0, 1, 2, 3, 8, 9, 10, 11]
    OPTICAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
    REQUIRED_SPECTRAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    STATIC_9_CHANNELS = [
        "Elevation",
        "Slope",
        "Northness",
        "Eastness",
        "Travel_Time_Roads",
        "Travel_Time_Trails",
        "Dist_to_Railways",
        "Dist_to_Camps",
        "Dist_to_Powerlines",
    ]
    ROAD_HIGHWAYS = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    }
    TRAIL_HIGHWAYS = {"track", "path", "footway"}
    OVERPASS_SERVERS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    def __init__(self):
        self._stac = None

    def _get_stac(self):
        if self._stac is None:
            import planetary_computer as pc
            import pystac_client

            self._stac = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace,
            )
        return self._stac

    @staticmethod
    def get_master_grid_info(lat: float, lon: float) -> dict:
        from pyproj import Transformer
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        zone = math.floor((lon + 180.0) / 6.0) + 1
        epsg_code = (32600 + zone) if lat >= 0 else (32700 + zone)
        utm_crs = CRS.from_epsg(epsg_code)

        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        cx, cy = to_utm.transform(lon, lat)
        min_x, max_x = cx - HALF_SCENE_METERS, cx + HALF_SCENE_METERS
        min_y, max_y = cy - HALF_SCENE_METERS, cy + HALF_SCENE_METERS
        transform = from_bounds(min_x, min_y, max_x, max_y, GRID_SIZE_PX, GRID_SIZE_PX)

        to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        corners = [to_wgs.transform(x, y) for x in (min_x, max_x) for y in (min_y, max_y)]
        lons = [p[0] for p in corners]
        lats = [p[1] for p in corners]
        return {
            "center_lat_lon": (lat, lon),
            "epsg": epsg_code,
            "crs": utm_crs,
            "transform": transform,
            "shape": (GRID_SIZE_PX, GRID_SIZE_PX),
            "pixel_scale_m": PIXEL_SCALE_METERS,
            "utm_bounds": (min_x, min_y, max_x, max_y),
            "bbox_wgs84": [round(min(lons), 5), round(min(lats), 5), round(max(lons), 5), round(max(lats), 5)],
        }

    @staticmethod
    def align_raster_to_master(
        src_array: np.ndarray,
        src_crs: Any,
        src_transform: Any,
        grid_info: dict,
        resampling_method: Any,
        src_nodata: Any = None,
        dst_nodata: float = np.nan,
    ) -> np.ndarray:
        from rasterio.warp import reproject

        dtype = np.float32 if np.isnan(dst_nodata) else src_array.dtype
        dst_array = np.full(grid_info["shape"], dst_nodata, dtype=dtype)
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=grid_info["transform"],
            dst_crs=grid_info["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling_method,
        )
        return dst_array

    def _read_band_window(self, asset_url: str, bbox_wgs84: List[float], is_categorical: bool = False):
        import planetary_computer as pc
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        try:
            signed_url = pc.sign(asset_url)
        except Exception:
            signed_url = asset_url

        for attempt in range(3):
            try:
                with rasterio.open(signed_url) as src:
                    west, south, east, north = bbox_wgs84
                    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
                    window = from_bounds(left, bottom, right, top, src.transform)
                    win_w = max(1.0, float(window.width))
                    win_h = max(1.0, float(window.height))
                    src_res_x = abs(src.transform.a) * (111320.0 if (src.crs and src.crs.is_geographic) else 1.0)
                    target_res = max(src_res_x, PIXEL_SCALE_METERS * 0.5)
                    scale_factor = max(1.0, target_res / max(src_res_x, 1e-6))
                    out_w = max(16, int(round(win_w / scale_factor)))
                    out_h = max(16, int(round(win_h / scale_factor)))
                    resamp = Resampling.nearest if is_categorical else Resampling.average
                    data = src.read(
                        1,
                        window=window,
                        out_shape=(out_h, out_w),
                        resampling=resamp,
                        boundless=True,
                        fill_value=0 if is_categorical else np.nan,
                    )
                    win_transform = src.window_transform(window) * Affine.scale(win_w / out_w, win_h / out_h)
                    return data.astype(np.float32), win_transform, src.crs
            except Exception:
                time.sleep(1.5 * (attempt + 1))
                try:
                    signed_url = pc.sign(asset_url)
                except Exception:
                    pass
        raise RuntimeError(f"Failed reading COG window from {asset_url[:80]}")

    def _fetch_mosaicked_band(self, band_name: str, items: List[Any], grid_info: dict):
        from rasterio.enums import Resampling

        is_scl = band_name == "SCL"
        resampling = Resampling.nearest if is_scl else Resampling.average
        shape = grid_info["shape"]
        canvas = np.zeros(shape, dtype=np.uint8) if is_scl else np.full(shape, np.nan, dtype=np.float32)

        for item in items:
            if band_name not in item.assets:
                continue
            try:
                raw_data, src_transform, src_crs = self._read_band_window(
                    item.assets[band_name].href, grid_info["bbox_wgs84"], is_categorical=is_scl
                )
                if not is_scl and band_name in self.OPTICAL_BANDS:
                    raw_data = np.where((raw_data > 0) & np.isfinite(raw_data), raw_data, np.nan)
                    baseline = str(item.properties.get("s2:processing_baseline", "02.00"))
                    if baseline >= "04.00":
                        raw_data = np.maximum(raw_data - 1000.0, 1.0)
                    raw_data = raw_data / 10000.0
                aligned = self.align_raster_to_master(
                    raw_data,
                    src_crs,
                    src_transform,
                    grid_info,
                    resampling,
                    dst_nodata=0.0 if is_scl else np.nan,
                )
                if is_scl:
                    m = aligned != 0
                    canvas[m] = aligned[m].astype(np.uint8)
                else:
                    m = np.isfinite(aligned) & (aligned > 0.0)
                    canvas[m] = aligned[m]
            except Exception:
                continue
        return band_name, canvas

    def fetch_s2_and_sar(self, grid_info: dict, pre_fire_date: str, lookback_days: int = 30, progress_cb=None) -> dict:
        target_dt = datetime.strptime(pre_fire_date, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)
        date_range = f"{start_dt.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}"
        stac = self._get_stac()

        # 1. Sentinel-2 L2A
        s2_items = list(
            stac.search(
                collections=["sentinel-2-l2a"],
                bbox=grid_info["bbox_wgs84"],
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": 65.0}},
            ).items()
        )
        if not s2_items:
            ext_start = target_dt - timedelta(days=60)
            ext_range = f"{ext_start.strftime('%Y-%m-%d')}/{target_dt.strftime('%Y-%m-%d')}"
            s2_items = list(
                stac.search(
                    collections=["sentinel-2-l2a"],
                    bbox=grid_info["bbox_wgs84"],
                    datetime=ext_range,
                ).items()
            )
        if not s2_items:
            raise RuntimeError(f"No Sentinel-2 scenes found in {date_range} for {grid_info['center_lat_lon']}")

        by_date: Dict[str, List[Any]] = defaultdict(list)
        for it in s2_items:
            by_date[it.datetime.strftime("%Y-%m-%d")].append(it)

        best_candidate = None  # (inv_ratio, cand_date, d_items, scl, loss_mask)
        cand_dates = sorted(by_date.keys(), reverse=True)[:6]

        for idx_d, cand_date in enumerate(cand_dates):
            if progress_cb:
                progress_cb(
                    0.22 + 0.06 * (idx_d / max(1, len(cand_dates))),
                    f"[2D Step 1/4] Checking Sentinel-2 cloud mask (SCL) for {cand_date}...",
                )
            d_items = by_date[cand_date]
            _, scl = self._fetch_mosaicked_band("SCL", d_items, grid_info)
            if scl is None:
                continue
            scl_clean = np.nan_to_num(scl, nan=0).astype(np.uint8)
            loss_mask = np.where(np.isin(scl_clean, self.SCL_SNOW), 0.0, np.isin(scl_clean, self.SCL_INVALID)).astype(
                np.float32
            )
            inv_ratio = float(loss_mask.mean())
            if best_candidate is None or inv_ratio < best_candidate[0]:
                best_candidate = (inv_ratio, cand_date, d_items, scl, loss_mask)
            if inv_ratio <= MAX_INVALID_RATIO:
                break

        if best_candidate is None:
            raise RuntimeError("No valid Sentinel-2 scene passed SCL check.")

        _, chosen_t0, chosen_items, chosen_scl, chosen_loss = best_candidate
        if progress_cb:
            progress_cb(0.30, f"[2D Step 1/4] Downloading 10 Sentinel-2 spectral bands for {chosen_t0}...")

        chosen_bands: Dict[str, np.ndarray] = {"SCL": chosen_scl}
        rem = [b for b in self.OPTICAL_BANDS if b != "SCL"]
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(self._fetch_mosaicked_band, b, chosen_items, grid_info) for b in rem]
            for f in as_completed(futs):
                bn, barr = f.result()
                chosen_bands[bn] = barr

        # 2. Sentinel-1 RTC SAR
        if progress_cb:
            progress_cb(0.40, "[2D Step 1/4] Querying Sentinel-1 RTC SAR backscatter (VV, VH)...")
        sar_bands: Dict[str, np.ndarray] = {}
        try:
            s1_items = list(
                stac.search(
                    collections=["sentinel-1-rtc"],
                    bbox=grid_info["bbox_wgs84"],
                    datetime=date_range,
                ).items()
            )
            s1_by_date: Dict[str, List[Any]] = defaultdict(list)
            for it in s1_items:
                s1_by_date[it.datetime.strftime("%Y-%m-%d")].append(it)
            for d_str in sorted(s1_by_date.keys(), reverse=True)[:4]:
                d_items = s1_by_date[d_str]
                _, vv = self._fetch_mosaicked_band("vv", d_items, grid_info)
                _, vh = self._fetch_mosaicked_band("vh", d_items, grid_info)
                if vv is not None and np.isfinite(vv).mean() >= 0.5 and vh is not None and np.isfinite(vh).mean() >= 0.5:
                    sar_bands = {"SAR_VV": vv, "SAR_VH": vh}
                    break
        except Exception as e:
            print(f"  [Sentinel-1 SAR] warning: {e}")

        return {
            "bands_t0": chosen_bands,
            "loss_mask": chosen_loss,
            "t0_date": chosen_t0,
            "sar_bands": sar_bands,
        }

    def fetch_static_terrain_and_osm(self, grid_info: dict, buffer_meters: float = 4000.0, progress_cb=None) -> Dict[str, np.ndarray]:
        """Fetches or loads cached 9 static channels: DEM (4) + OSM accessibility/distances (5)."""
        lat, lon = grid_info["center_lat_lon"]
        cache_dir = Path(CONFIG["MINI_DATASET_DIR"]) / "static_2d_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"static_{lat:.4f}_{lon:.4f}.npz"

        if cache_file.exists():
            try:
                with np.load(cache_file) as z:
                    if all(k in z.files for k in self.STATIC_9_CHANNELS):
                        if progress_cb:
                            progress_cb(0.60, "[2D Step 2/4] Loaded cached Copernicus DEM & OSM travel-time rasters...")
                        return {k: z[k].astype(np.float32) for k in self.STATIC_9_CHANNELS}
            except Exception:
                pass

        import planetary_computer as pc
        from pyproj import Transformer
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.features import rasterize
        from rasterio.merge import merge
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject

        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        b_min_x, b_max_x = min_x - buffer_meters, max_x + buffer_meters
        b_min_y, b_max_y = min_y - buffer_meters, max_y + buffer_meters
        res_50m = 50.0
        w_50 = int(round((b_max_x - b_min_x) / res_50m))
        h_50 = int(round((b_max_y - b_min_y) / res_50m))
        buf_shape_50m = (h_50, w_50)
        buf_transform_50m = from_bounds(b_min_x, b_min_y, b_max_x, b_max_y, w_50, h_50)

        to_wgs = Transformer.from_crs(grid_info["crs"], "EPSG:4326", always_xy=True)
        corners = [to_wgs.transform(x, y) for x in (b_min_x, b_max_x) for y in (b_min_y, b_max_y)]
        west, east = min(p[0] for p in corners), max(p[0] for p in corners)
        south, north = min(p[1] for p in corners), max(p[1] for p in corners)
        padded_bbox = (west - 0.01, south - 0.01, east + 0.01, north + 0.01)

        # 1. Copernicus DEM 30m
        if progress_cb:
            progress_cb(0.48, "[2D Step 2/4] Fetching Copernicus DEM GLO-30 elevation, slope, and aspect...")
        stac = self._get_stac()
        dem_items = list(stac.search(collections=["cop-dem-glo-30"], bbox=padded_bbox).items())
        if not dem_items:
            raise RuntimeError(f"No Copernicus DEM tiles found for {padded_bbox}")

        src_files = [rasterio.open(pc.sign(it.assets["data"].href)) for it in dem_items]
        try:
            mosaic_arr, mosaic_transform = merge(src_files, bounds=padded_bbox)
            src_crs = src_files[0].crs
            src_nodata = src_files[0].nodata
        finally:
            for s in src_files:
                try:
                    s.close()
                except Exception:
                    pass

        dem_50m = np.full(buf_shape_50m, np.nan, dtype=np.float32)
        reproject(
            source=mosaic_arr[0],
            destination=dem_50m,
            src_transform=mosaic_transform,
            src_crs=src_crs,
            dst_transform=buf_transform_50m,
            dst_crs=grid_info["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )

        px, py = abs(buf_transform_50m.a), abs(buf_transform_50m.e)
        dz_dy, dz_dx = np.gradient(dem_50m, py, px)
        dz_dnorth = -dz_dy
        slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dnorth**2))).astype(np.float32)
        aspect_rad = np.arctan2(-dz_dx, -dz_dnorth) % (2.0 * np.pi)
        northness = np.cos(aspect_rad).astype(np.float32)
        eastness = np.sin(aspect_rad).astype(np.float32)
        flat = slope_deg < 0.1
        northness[flat] = 0.0
        eastness[flat] = 0.0

        results: Dict[str, np.ndarray] = {}
        for name, arr in (
            ("Elevation", dem_50m),
            ("Slope", slope_deg),
            ("Northness", northness),
            ("Eastness", eastness),
        ):
            results[name] = self.align_raster_to_master(
                arr, grid_info["crs"], buf_transform_50m, grid_info, Resampling.average, dst_nodata=np.nan
            )

        # 2. OSM Vectors via Overpass API
        if progress_cb:
            progress_cb(0.55, "[2D Step 2/4] Fetching OpenStreetMap roads, trails, powerlines, and camps via Overpass...")
        gdf_utm = self._fetch_overpass_osm(west, south, east, north, grid_info["crs"])

        def _rast(df_sub):
            if df_sub is None or df_sub.empty or "geometry" not in df_sub.columns:
                return np.zeros(buf_shape_50m, dtype=np.uint8)
            shapes = [(g, 1) for g in df_sub["geometry"] if g is not None and not g.is_empty]
            if not shapes:
                return np.zeros(buf_shape_50m, dtype=np.uint8)
            return rasterize(shapes, out_shape=buf_shape_50m, transform=buf_transform_50m, fill=0, dtype=np.uint8, all_touched=True)

        gdf_roads = gdf_trails = gdf_water = gdf_bridges = gdf_rail = gdf_camps = gdf_power = None
        if gdf_utm is not None and not gdf_utm.empty:
            if "highway" in gdf_utm.columns:
                gdf_roads = gdf_utm[gdf_utm["highway"].isin(self.ROAD_HIGHWAYS)]
                gdf_trails = gdf_utm[gdf_utm["highway"].isin(self.TRAIL_HIGHWAYS)]
            if "natural" in gdf_utm.columns or "waterway" in gdf_utm.columns:
                wc = pd.Series(False, index=gdf_utm.index)
                if "natural" in gdf_utm.columns:
                    wc = wc | gdf_utm["natural"].isin(["water", "wetland"])
                if "waterway" in gdf_utm.columns:
                    wc = wc | gdf_utm["waterway"].isin(["river", "stream", "canal", "riverbank"])
                gdf_water = gdf_utm[wc]
            if "bridge" in gdf_utm.columns:
                gdf_bridges = gdf_utm[gdf_utm["bridge"].notna() & (gdf_utm["bridge"] != "no")]
            if "railway" in gdf_utm.columns:
                gdf_rail = gdf_utm[gdf_utm["railway"].isin(["rail", "narrow_gauge", "spur"])]
            if "power" in gdf_utm.columns:
                gdf_power = gdf_utm[gdf_utm["power"].isin(["line", "minor_line", "cable", "substation"])]
            if "tourism" in gdf_utm.columns or "amenity" in gdf_utm.columns:
                cc = pd.Series(False, index=gdf_utm.index)
                if "tourism" in gdf_utm.columns:
                    cc = cc | gdf_utm["tourism"].isin(["camp_site", "picnic_site", "wilderness_hut"])
                if "amenity" in gdf_utm.columns:
                    cc = cc | gdf_utm["amenity"].isin(["shelter", "firepit"])
                gdf_camps = gdf_utm[cc]

        if progress_cb:
            progress_cb(0.60, "[2D Step 2/4] Computing Dijkstra travel time and distance transforms...")
        passable_50m = np.ones(buf_shape_50m, dtype=np.uint8)
        passable_50m[_rast(gdf_water) == 1] = 0
        passable_50m[np.isnan(dem_50m)] = 0
        passable_50m[_rast(gdf_bridges) == 1] = 1

        for key_name, gdf_sub in (("Travel_Time_Roads", gdf_roads), ("Travel_Time_Trails", gdf_trails)):
            src_mask = _rast(gdf_sub)
            src_mask[passable_50m == 0] = 0
            t50 = _dijkstra_travel_time_50m(dem_50m, src_mask, passable_50m, resolution=50.0, max_time_cap_hours=12.0)
            results[key_name] = self.align_raster_to_master(
                t50, grid_info["crs"], buf_transform_50m, grid_info, Resampling.average, dst_nodata=12.0
            )

        for key_name, gdf_sub in (
            ("Dist_to_Railways", gdf_rail),
            ("Dist_to_Camps", gdf_camps),
            ("Dist_to_Powerlines", gdf_power),
        ):
            m50 = _rast(gdf_sub)
            if np.any(m50 == 1):
                d50 = (distance_transform_edt(1 - m50) * 50.0).astype(np.float32)
                results[key_name] = self.align_raster_to_master(
                    d50, grid_info["crs"], buf_transform_50m, grid_info, Resampling.average, dst_nodata=50000.0
                )
            else:
                results[key_name] = np.full(grid_info["shape"], 50000.0, dtype=np.float32)

        try:
            np.savez_compressed(cache_file, **results)
        except Exception:
            pass
        return results

    def _fetch_overpass_osm(self, west: float, south: float, east: float, north: float, target_crs: Any) -> pd.DataFrame:
        from pyproj import Transformer
        from shapely.geometry import LineString, Point, Polygon

        empty_df = pd.DataFrame(
            columns=["geometry", "highway", "railway", "power", "natural", "waterway", "tourism", "amenity", "bridge"]
        )
        to_utm = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

        query = f"""
        [out:json][timeout:25];
        (
          way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|path|footway|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          way["railway"~"rail|narrow_gauge|spur"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          way["power"~"line|minor_line|cable|substation"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          way["natural"~"water|wetland"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          way["waterway"~"river|stream|canal|riverbank"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          node["tourism"~"camp_site|picnic_site|wilderness_hut"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          way["tourism"~"camp_site|picnic_site|wilderness_hut"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          node["amenity"~"shelter|firepit"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
          way["amenity"~"shelter|firepit"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
        );
        out body geom;
        """
        sess = _make_session()
        for url in self.OVERPASS_SERVERS:
            try:
                r = sess.post(url, data={"data": query}, timeout=30)
                if r.status_code != 200:
                    continue
                elements = r.json().get("elements", [])
                rows = []
                for el in elements:
                    tags = el.get("tags", {})
                    geom = None
                    if el.get("type") == "node" and "lat" in el and "lon" in el:
                        x_u, y_u = to_utm.transform(float(el["lon"]), float(el["lat"]))
                        geom = Point(x_u, y_u)
                    elif "geometry" in el and len(el["geometry"]) >= 2:
                        lons = [float(pt["lon"]) for pt in el["geometry"]]
                        lats = [float(pt["lat"]) for pt in el["geometry"]]
                        xs, ys = to_utm.transform(lons, lats)
                        coords = list(zip(xs, ys))
                        if (
                            len(coords) >= 4
                            and coords[0] == coords[-1]
                            and ("natural" in tags or "waterway" in tags or "tourism" in tags)
                        ):
                            try:
                                geom = Polygon(coords)
                            except Exception:
                                geom = LineString(coords)
                        else:
                            geom = LineString(coords)
                    if geom is not None and not geom.is_empty:
                        rows.append(
                            {
                                "geometry": geom,
                                "highway": tags.get("highway"),
                                "railway": tags.get("railway"),
                                "power": tags.get("power"),
                                "natural": tags.get("natural"),
                                "waterway": tags.get("waterway"),
                                "tourism": tags.get("tourism"),
                                "amenity": tags.get("amenity"),
                                "bridge": tags.get("bridge"),
                            }
                        )
                if not rows:
                    return empty_df
                return pd.DataFrame(rows)
            except Exception as e:
                print(f"  [Overpass] mirror {url} failed ({e}), trying next...")
        return empty_df

    def fetch_soil_moisture(self, grid_info: dict, pre_fire_date: str, default_moisture: float = 0.20) -> np.ndarray:
        """Fetches ERA5-Land top-layer volumetric soil moisture (0-7 cm).
        Tries GEE first if configured; falls back to Open-Meteo `soil_moisture_0_to_7cm` (no key needed!)."""
        shape = grid_info["shape"]
        lat, lon = grid_info["center_lat_lon"]

        # 1. Try Open-Meteo ERA5-Land soil_moisture_0_to_7cm (fast, free, no GEE key required)
        try:
            dt = datetime.strptime(pre_fire_date, "%Y-%m-%d").date()
            recent = (date.today() - dt).days < 5
            url = "https://api.open-meteo.com/v1/forecast" if recent else "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "start_date": (dt - timedelta(days=3)).isoformat(),
                "end_date": dt.isoformat(),
                "hourly": "soil_moisture_0_to_7cm",
                "timezone": "UTC",
            }
            r = _make_session().get(url, params=params, timeout=20)
            if r.status_code == 200:
                vals = [x for x in r.json().get("hourly", {}).get("soil_moisture_0_to_7cm", []) if x is not None]
                if vals:
                    sm_val = float(np.clip(np.mean(vals[-24:]), 0.0, 1.0))
                    return np.full(shape, sm_val, dtype=np.float32)
        except Exception:
            pass

        return np.full(shape, default_moisture, dtype=np.float32)

    def collect_full_scene(
        self,
        lat: float,
        lon: float,
        target_day: date,
        weather_dict: Optional[Dict[str, float]] = None,
        pop_10_50: Optional[Tuple[float, float]] = None,
        progress_cb=None,
    ) -> Dict[str, Any]:
        """Collects the complete 22-channel 128x128 2D raster + SCL + 35-dim 1D context vector for (lat, lon, target_day)."""
        t0_wall = time.perf_counter()
        grid_info = self.get_master_grid_info(lat, lon)
        wd = target_day - timedelta(days=1)
        pre_fire_str = wd.isoformat()

        if progress_cb:
            progress_cb(0.20, "[2D Step 1/4] Querying Planetary Computer STAC for Sentinel-2 L2A & Sentinel-1 SAR...")
        sent_res = self.fetch_s2_and_sar(grid_info, pre_fire_str, 30, progress_cb=progress_cb)

        if progress_cb:
            progress_cb(0.46, "[2D Step 2/4] Fetching Copernicus DEM GLO-30 & OpenStreetMap roads (Dijkstra travel time)...")
        stat_res = self.fetch_static_terrain_and_osm(grid_info, 4000.0, progress_cb=progress_cb)

        if progress_cb:
            progress_cb(0.62, "[2D Step 3/4] Fetching ERA5-Land volumetric soil moisture (0-7 cm)...")
        soil_arr = self.fetch_soil_moisture(grid_info, pre_fire_str)

        t0_date_str = sent_res["t0_date"]
        t0_dt = datetime.strptime(t0_date_str, "%Y-%m-%d").date()

        if weather_dict is None:
            weather_dict = fetch_weather(lat, lon, wd, t0_dt)
        if pop_10_50 is not None:
            p10, p50 = pop_10_50
        else:
            p10, p50 = fetch_population_single(lat, lon, target_day.year)
        ctx_dict = {
            **weather_dict,
            "population_10km_log1p": float(p10),
            "population_50km_log1p": float(p50),
        }

        rasters_2d = {
            **{k: v for k, v in sent_res["bands_t0"].items() if k != "SCL"},
            **sent_res["sar_bands"],
            **stat_res,
            "Soil_Moisture": soil_arr,
        }

        h, w = grid_info["shape"]
        canonical_list = []
        for ch in CHANNELS_2D:
            arr = rasters_2d.get(ch)
            if arr is not None:
                canonical_list.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            else:
                canonical_list.append(np.zeros((h, w), dtype=np.float32))

        X_2d = np.stack(canonical_list, axis=0).astype(np.float32)  # (22, 128, 128)
        scl = sent_res["bands_t0"]["SCL"].astype(np.uint8)  # (128, 128)
        X_1d = raw_context_vector(ctx_dict)  # (35,)

        has_sar = float("SAR_VV" in sent_res["sar_bands"] and "SAR_VH" in sent_res["sar_bands"])
        channel_validity = np.array(
            [has_sar if ch in ("SAR_VV", "SAR_VH") else 1.0 for ch in CHANNELS_2D],
            dtype=np.float32,
        )
        scene_id = f"SCENE_{lat:.4f}_{lon:.4f}_{target_day.strftime('%Y%m%d')}"
        patch_id = f"{scene_id}_p000"
        valid_s2_frac = float(1.0 - (sent_res["loss_mask"] > 0.5).mean())

        return {
            "scene_id": scene_id,
            "patch_id": patch_id,
            "lat": float(lat),
            "lon": float(lon),
            "latitude": float(lat),
            "longitude": float(lon),
            "date": target_day.isoformat(),
            "target_date": target_day.isoformat(),
            "t0_date": t0_date_str,
            "s2_Source_Date": t0_date_str,
            "valid_s2_frac": valid_s2_frac,
            "crs": str(grid_info["crs"]),
            "invalid_ratio": float((sent_res["loss_mask"] > 0.5).mean()),
            "sar_coverage_ratio": float(has_sar),
            "has_sar": int(has_sar > 0.5),
            "X_2d": X_2d,
            "scl": scl,
            "X_1d": X_1d,
            "channel_validity": channel_validity,
            "context_dict": ctx_dict,
            "elapsed_sec": time.perf_counter() - t0_wall,
        }

    collect_scene = collect_full_scene

