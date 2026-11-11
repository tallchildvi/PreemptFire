"""Canadian Fire Weather Index from the EFFIS WMS (Copernicus EMS / JRC) instead of a home-made reconstruction.

One GetMap request returns the FWI field for the whole Iberian bounding box on one date, so a map costs 1 request
(the previous version needed ~300 Open-Meteo requests plus an approximate reconstruction). Rasters are cached on disk.

Environment variables (all optional):
  EFFIS_WMS_URL    default https://maps.effis.emergency.copernicus.eu/effis
  EFFIS_FWI_LAYER  default ecmwf007.fwi   (check with `python fwi_calculator.py layers`)
  EFFIS_CACHE_DIR  default effis_cache
  EFFIS_WORKERS    default 3

Command line:
  python fwi_calculator.py selfcheck [YYYY-MM-DD]       fetch one day and print what came back (run this first)
  python fwi_calculator.py layers [text]                list layers/time extents from GetCapabilities
  python fwi_calculator.py oof oof_predictions.csv      FWI vs your models on the same scenes (paired block bootstrap)
"""
from __future__ import annotations

import hashlib, io, json, math, os, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

WMS_URL = os.getenv("EFFIS_WMS_URL", "https://maps.effis.emergency.copernicus.eu/gwis")
FWI_LAYER = os.getenv("EFFIS_FWI_LAYER", "ecmwf.fwi")
CACHE_DIR = Path(os.getenv("EFFIS_CACHE_DIR", "effis_cache"))
WORKERS = int(os.getenv("EFFIS_WORKERS", "3"))
BBOX = (-9.7, 35.7, 3.6, 44.0)          # lon_min, lat_min, lon_max, lat_max (Iberia incl. a margin)
RES_DEG = 0.05                           # request grid; the source is ~8 km (~0.07 deg), the server resamples
EFFIS_CLASSES = ((5.2, "Very low"), (11.2, "Low"), (21.3, "Moderate"), (38.0, "High"), (50.0, "Very high"), (1e9, "Extreme"))
DEFAULT_TIERS = (11.2, 21.3)             # fallback Low / Medium / High cut points on the official EFFIS scale
CAL_FILE = CACHE_DIR / "fwi_calibration.json"


class EffisError(RuntimeError):
    pass


# ------------------------------------------------------------------ HTTP
_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                                                          respect_retry_after_header=True), pool_maxsize=max(4, WORKERS)))
        s.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])))
        s.headers["User-Agent"] = "PreemptFire-research/1.0"
        _SESSION = s
    return _SESSION


def _as_date(d) -> date:
    return d if isinstance(d, date) and not isinstance(d, datetime) else (d.date() if isinstance(d, datetime) else datetime.strptime(str(d)[:10], "%Y-%m-%d").date())


def list_layers(text: str = "") -> List[dict]:
    """Layers from GetCapabilities with their time extent (use it to pick EFFIS_FWI_LAYER)."""
    import xml.etree.ElementTree as ET
    out = []
    seen = set()
    for url in dict.fromkeys([WMS_URL, "https://maps.effis.emergency.copernicus.eu/gwis", "https://maps.effis.emergency.copernicus.eu/effis"]):
        try:
            r = _session().get(url, params={"SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetCapabilities"}, timeout=60)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for lay in root.iter("Layer"):
                name = lay.findtext("Name")
                if not name or (url, name) in seen:
                    continue
                seen.add((url, name))
                ext = lay.find("Extent")
                if ext is None:
                    dim = lay.find("Dimension")
                    ext_text = (dim.text or "") if dim is not None and (dim.get("name") or "").lower() == "time" else ""
                else:
                    ext_text = ext.text or ""
                out.append({"wms_url": url, "name": name, "title": lay.findtext("Title") or "", "time": ext_text.strip()[:120]})
        except Exception:
            continue
    return [x for x in out if text.lower() in (x["name"] + " " + x["title"]).lower()]


# ------------------------------------------------------------------ raster download and decoding
def _decode(content: bytes) -> np.ndarray:
    head = content[:200].lstrip()
    if head.startswith(b"<"):
        raise EffisError("EFFIS returned an XML error instead of an image: " + content[:400].decode("utf8", "replace"))
    arr, nodata = None, None
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with MemoryFile(content) as mf, mf.open() as ds:
                if ds.count >= 3 or ds.dtypes[0] == "uint8":
                    raise EffisError("EFFIS returned a colour-rendered image (%d bands, %s), not FWI values. "
                                     "Another FORMAT/layer is needed (see the self-check)." % (ds.count, ds.dtypes[0]))
                arr, nodata = ds.read(1).astype("float32"), ds.nodata
    except EffisError:
        raise
    except Exception:
        arr = None
    if arr is None:
        from PIL import Image
        im = Image.open(io.BytesIO(content))
        if im.mode in ("RGB", "RGBA", "P", "L"):
            raise EffisError("EFFIS returned a colour-rendered image (mode %s), not FWI values." % im.mode)
        arr = np.array(im, dtype="float32")
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    arr[(arr < -9000) | (arr > 1e5)] = np.nan
    # MapServer float32 GeoTIFFs encode ocean/background pixels as exact 0.0 without a GDAL nodata tag
    if nodata is None and np.isfinite(arr).any() and (arr == 0.0).mean() > 0.10 and (arr > 0.0).any():
        arr[arr == 0.0] = np.nan
    return arr


def _size(bbox=BBOX, res=RES_DEG) -> Tuple[int, int]:
    return int(round((bbox[3] - bbox[1]) / res)), int(round((bbox[2] - bbox[0]) / res))    # H, W


def _candidate_endpoints(wms_url: str, layer: str) -> List[Tuple[str, str]]:
    """Returns candidate (wms_url, layer) pairs on Copernicus EFFIS/GWIS WMS in case legacy layer names are configured."""
    suffix = layer.rsplit(".", 1)[-1] if "." in layer else "fwi"
    cands = [
        (wms_url, layer),
        ("https://maps.effis.emergency.copernicus.eu/gwis", f"ecmwf.{suffix}"),
        ("https://maps.effis.emergency.copernicus.eu/gwis", f"nasa_geos5.{suffix}"),
        ("https://maps.effis.emergency.copernicus.eu/effis", layer),
    ]
    out: List[Tuple[str, str]] = []
    for pair in cands:
        if pair not in out:
            out.append(pair)
    return out


def fetch_raster(day, layer: Optional[str] = None, bbox=BBOX, res: float = RES_DEG, use_cache: bool = True) -> np.ndarray:
    """FWI (or another EFFIS layer) for one date on the request grid; rows go from north to south. One HTTP request, cached."""
    day, req_layer = _as_date(day), layer or FWI_LAYER
    H, W = _size(bbox, res)

    last_err: Optional[Exception] = None
    for cand_url, cand_layer in _candidate_endpoints(WMS_URL, req_layer):
        key = hashlib.md5(f"{cand_url}|{cand_layer}|{day}|{bbox}|{H}x{W}".encode()).hexdigest()[:16]
        path = CACHE_DIR / f"{cand_layer.replace('/', '_')}_{day}_{key}.npy"
        if use_cache and path.exists():
            try:
                cached = np.load(path)
                if cached.shape == (H, W) and not np.isnan(cached).all():
                    return cached
            except Exception:
                pass
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": cand_layer,
            "STYLES": "",
            "FORMAT": "image/tiff",
            "SRS": "EPSG:4326",
            "BBOX": ",".join(f"{v:.4f}" for v in bbox),
            "WIDTH": W,
            "HEIGHT": H,
            "TIME": day.isoformat(),
        }
        try:
            r = _session().get(cand_url, params=params, timeout=90)
            if r.status_code != 200:
                raise EffisError(f"EFFIS HTTP {r.status_code} for {cand_layer} {day}: {r.text[:200]}")
            arr = _decode(r.content)
            if arr.shape != (H, W):
                raise EffisError(f"unexpected raster shape {arr.shape}, wanted {(H, W)}")
            if np.isnan(arr).all():
                raise EffisError(f"EFFIS returned an empty raster for {cand_layer} {day} (no data for that date?)")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
            return arr
        except Exception as exc:
            last_err = exc
            continue

    raise EffisError(str(last_err) if last_err else f"EFFIS fetch failed for {req_layer} {day}")


def fetch_many(days: Sequence, layer: Optional[str] = None, workers: int = WORKERS) -> Dict[date, Optional[np.ndarray]]:
    """Rasters for many dates in parallel. A failed date gives None and a printed reason instead of stopping the batch."""
    days = sorted({_as_date(d) for d in days})

    def one(d):
        for attempt in range(2):
            try:
                return d, fetch_raster(d, layer)
            except Exception as e:
                last = e
                time.sleep(2 + attempt * 3)
        print(f"  [EFFIS] {d}: {last}")
        return d, None
    with ThreadPoolExecutor(max(1, workers)) as ex:
        return dict(ex.map(one, days))


# ------------------------------------------------------------------ sampling
def _fill_nan(a: np.ndarray) -> np.ndarray:
    m = np.isnan(a)
    if not m.any():
        return a
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _neighbourhood(a: np.ndarray, radius_km: float, stat: str, res: float = RES_DEG, lat0: float = 40.0) -> np.ndarray:
    from scipy import ndimage
    dy, dx = radius_km / 111.0 / res, radius_km / (111.0 * math.cos(math.radians(lat0))) / res
    ry, rx = max(0, int(round(dy))), max(0, int(round(dx)))
    if ry == 0 and rx == 0:
        return a
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    fp = ((yy / max(dy, 1e-6)) ** 2 + (xx / max(dx, 1e-6)) ** 2) <= 1.0
    if stat == "max":
        return ndimage.maximum_filter(a, footprint=fp, mode="nearest")
    return ndimage.convolve(a, fp / fp.sum(), mode="nearest")


def sample(raster: np.ndarray, lat, lon, bbox=BBOX, radius_km: float = 0.0, stat: str = "mean") -> np.ndarray:
    a = _fill_nan(raster)
    if radius_km > 0:
        a = _neighbourhood(a, radius_km, stat, (bbox[2] - bbox[0]) / raster.shape[1])
    H, W = a.shape
    col = np.clip(((np.asarray(lon, float) - bbox[0]) / (bbox[2] - bbox[0]) * W).astype(int), 0, W - 1)
    row = np.clip(((bbox[3] - np.asarray(lat, float)) / (bbox[3] - bbox[1]) * H).astype(int), 0, H - 1)
    return a[row, col].astype(np.float32)


def fwi_for_points(lat, lon, day, layer: Optional[str] = None, radius_km: float = 0.0, stat: str = "mean") -> np.ndarray:
    return sample(fetch_raster(day, layer), lat, lon, radius_km=radius_km, stat=stat)


def effis_class(fwi: float) -> str:
    return next(name for cut, name in EFFIS_CLASSES if fwi < cut)


def dsr_from_fwi(fwi):
    return 0.0272 * np.maximum(np.asarray(fwi, float), 0.0) ** 1.77


# ------------------------------------------------------------------ calibration (FWI -> probability) and tier cut points
_CAL: Optional[dict] = None


def _load_cal() -> Optional[dict]:
    global _CAL
    if _CAL is None and CAL_FILE.exists():
        _CAL = json.load(open(CAL_FILE))
    return _CAL


def fit_calibration(fwi: np.ndarray, y: np.ndarray, shares=(0.6, 0.3)) -> dict:
    """Isotonic map FWI -> P(fire within 50 km) fitted on labelled scenes, plus tier cut points from the same scenes.
    Fitted in-sample: use it for map colours, not for reporting skill (skill is the AUC, which needs no calibration)."""
    from sklearn.isotonic import IsotonicRegression
    ok = np.isfinite(fwi)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=0.999).fit(fwi[ok], y[ok])
    cal = {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_],
           "tiers": [float(v) for v in np.quantile(fwi[ok], [shares[0], shares[0] + shares[1]])], "n": int(ok.sum())}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(cal, open(CAL_FILE, "w"))
    global _CAL
    _CAL = cal
    return cal


def fwi_to_p(fwi: np.ndarray) -> Tuple[np.ndarray, bool]:
    cal = _load_cal()
    if cal:
        return np.interp(fwi, cal["x"], cal["y"]).astype(np.float32), True
    return np.clip(np.asarray(fwi, float) / (np.asarray(fwi, float) + 30.0), 1e-4, 0.999).astype(np.float32), False   # monotone score, NOT a probability


def tier_cuts() -> Tuple[float, float]:
    cal = _load_cal()
    return tuple(cal["tiers"]) if cal else DEFAULT_TIERS


def tier_cuts_p() -> Optional[Tuple[float, float]]:
    """The same tier cut points expressed on the calibrated probability scale (for get_thresholds in the registry); None if not calibrated."""
    cal = _load_cal()
    return tuple(float(v) for v in np.interp(cal["tiers"], cal["x"], cal["y"])) if cal else None


# ------------------------------------------------------------------ interface used by the Space
def compute_fwi_for_grid(lat, lon, day, radius_km: float = 0.0, stat: str = "mean", components: bool = False) -> Dict[str, np.ndarray]:
    """FWI at every grid point for `day` (one request; one more per extra component). Keys follow the old module."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    fwi = fwi_for_points(lat, lon, day, radius_km=radius_km, stat=stat)
    out = {"fwi": fwi, "dsr": dsr_from_fwi(fwi).astype(np.float32)}
    for k in ("ffmc", "dmc", "dc", "isi", "bui"):
        out[k] = np.full(len(fwi), np.nan, np.float32)
    if components:
        prefix = FWI_LAYER.rsplit(".", 1)[0]
        for k in ("ffmc", "dmc", "dc", "isi", "bui"):
            try:
                out[k] = fwi_for_points(lat, lon, day, layer=f"{prefix}.{k}", radius_km=radius_km, stat=stat)
            except Exception as e:
                print(f"  [EFFIS] component {k} unavailable: {e}")
    out["p_fwi"], out["p_is_calibrated"] = fwi_to_p(fwi)
    return out


_CTX: dict = {}


def set_scoring_context(day, lat_lon) -> None:
    """Call once before predicting: the real scoring date and the (n, 2) array of grid lat/lon."""
    _CTX["day"], _CTX["lat_lon"] = _as_date(day), np.asarray(lat_lon, float)


def compute_fwi_for_grid_from_tab1d(TAB_1D, col_names=None, weather_day=None, pts_lat_lon=None, day=None) -> Dict[str, np.ndarray]:
    """Compatibility wrapper for model_registry. The weather columns are not used any more; the scoring context is."""
    target_day = _as_date(day) if day is not None else (_CTX.get("day") or (_as_date(weather_day) if weather_day is not None else None))
    ll = np.asarray(pts_lat_lon, float) if pts_lat_lon is not None else _CTX.get("lat_lon")
    if target_day is None or ll is None:
        raise EffisError("call fwi_calculator.set_scoring_context(day, pts_lat_lon) before predicting with the fwi model")
    if len(ll) != len(TAB_1D):
        raise EffisError(f"context has {len(ll)} points but TAB_1D has {len(TAB_1D)}")
    return compute_fwi_for_grid(ll[:, 0], ll[:, 1], target_day)


# ------------------------------------------------------------------ evaluation on the scenes of the notebook
def scene_fwi(df, lag_days: int = 0, radius_km: float = 0.0, stat: str = "mean", layer: Optional[str] = None) -> np.ndarray:
    """FWI for every row of a frame with lat, lon, date. One request per distinct date (cached)."""
    d = np.array([_as_date(x) - timedelta(days=lag_days) for x in df["date"]])
    rasters = fetch_many(set(d), layer)
    out = np.full(len(df), np.nan, np.float32)
    for day in set(d):
        ras = rasters.get(day)
        if ras is None:
            continue
        m = d == day
        out[m] = sample(ras, df["lat"].values[m], df["lon"].values[m], radius_km=radius_km, stat=stat)
    return out


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def compare_on_oof(oof_csv: str, models=("tab_clim_full", "persist", "clim"), n_boot: int = 1000, seed: int = 0, radius_km: float = 0.0):
    import pandas as pd
    df = pd.read_csv(oof_csv)
    for lag in (0, 1):
        df[f"p_fwi_lag{lag}"] = scene_fwi(df, lag, radius_km)
    if radius_km <= 0:
        df["p_fwi_max50"] = scene_fwi(df, 0, 50.0, "max")
    cols = [c for c in df.columns if c.startswith("p_fwi")] + [f"p_{m}" for m in models if f"p_{m}" in df.columns]
    ok = df[[c for c in cols]].notna().all(axis=1).values
    d = df[ok].reset_index(drop=True)
    print(f"scenes with FWI: {len(d)}/{len(df)}")
    y, g = d["y"].values.astype(int), d["group"].values
    rng = np.random.default_rng(seed)
    idx_by = {k: np.where(g == k)[0] for k in np.unique(g)}
    keys = list(idx_by)
    draws = [np.concatenate([idx_by[k] for k in rng.choice(keys, len(keys))]) for _ in range(n_boot)]
    rows = []
    for c in cols:
        pt = _auc(y, d[c].values)
        bs = np.array([_auc(y[i], d[c].values[i]) for i in draws])
        rows.append({"score": c, "AUC": pt, "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5)})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    base = "p_fwi_lag0"
    for m in [c for c in cols if not c.startswith("p_fwi")]:
        diff = np.array([_auc(y[i], d[m].values[i]) - _auc(y[i], d[base].values[i]) for i in draws])
        print(f"  {m} minus {base}: {_auc(y, d[m].values) - _auc(y, d[base].values):+.3f}  95% CI [{np.nanpercentile(diff, 2.5):+.3f}, {np.nanpercentile(diff, 97.5):+.3f}]")
    d.to_csv("oof_with_fwi.csv", index=False); res.to_csv("fwi_vs_models.csv", index=False)
    fit_calibration(d[base].values, y)
    print("saved oof_with_fwi.csv, fwi_vs_models.csv and the calibration/tier file", CAL_FILE)
    return res


def selfcheck(day: str = "2025-08-12") -> None:
    ras = fetch_raster(day, use_cache=False)
    print(f"{FWI_LAYER} {day}: shape={ras.shape} dtype={ras.dtype} min={np.nanmin(ras):.1f} max={np.nanmax(ras):.1f} median={np.nanmedian(ras):.1f} nan={np.isnan(ras).mean():.0%}")
    for name, la, lo in (("Madrid", 40.42, -3.70), ("Sevilla", 37.39, -5.98), ("Santiago", 42.88, -8.54), ("Pyrenees", 42.6, 0.5)):
        v = float(sample(ras, [la], [lo])[0])
        print(f"  {name:9s} FWI={v:6.1f} ({effis_class(v)})")
    print("Plausible in August: Madrid/Sevilla tens of units, Santiago lower. All zeros, 0-255 integers or one constant value mean the layer/format is wrong.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "layers":
        for x in list_layers(sys.argv[2] if len(sys.argv) > 2 else "fwi"):
            print(x)
    elif cmd == "oof":
        compare_on_oof(sys.argv[2])
    else:
        selfcheck(sys.argv[2] if len(sys.argv) > 2 else "2025-08-12")