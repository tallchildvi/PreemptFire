"""Кандидати точок для гілки 100 м / 128x128 (12.8 км) — сітка ТІЛЬКИ по території Іспанії та Португалії.

Принципи:
  * територія: полігони ESP + PRT (ті самі geo.json, що в fire_date.py), обрізані bbox Іберії -> без Франції, Марокко, Андорри, моря;
    сцена береться, якщо центр у країні і >= MIN_COUNTRY_FRACTION квадрата 12.8 км лежить у країні;
  * кожна хвиля — та сама сітка, зсунута на частку кроку (хвиля 1: 0/0, хвиля 2: 0.5/0.5, хвиля 3: 0.5/0, хвиля 4: 0/0.5),
    тож позиції НЕ повторюються і територія покривається густіше; одна дата на локацію;
  * роки 2017–2025 розподілені порівну (EFFIS-шари з 2016 => для 2017+ є повний рік історії; для 2017 лише з липня — S2B);
  * сезон: червень–вересень + частка «плечей» (березень–травень, жовтень);
  * вихід перемішаний; зупинка збору в будь-який момент лишає репрезентативну підвибірку;
  * друкує, яку частку території ESP+PRT покривають сцени (по хвилях і разом) і зберігає карту PNG.
Вихід (INTERIM_DATA_DIR): points_grid_100m.csv (усі), ..._wave{1..N}.csv, points_grid_100m_map.png.
Колонки: lat, lon, target_date, is_fire(=0), loc_id, wave.
Запуск:  python -m src.data_pipeline.make_grid_points_100m
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

IBERIA = (35.9, -9.5, 43.9, 3.4)      # min_lat, min_lon, max_lat, max_lon (відсікає Канари, Азори, Мадейру)
SCENE_KM = 12.8
STEP_DEG = 0.115                       # ~12.8 км: сцени сусідніх вузлів майже стикуються без перекриття
JITTER_DEG = 0.008                      # ~0.9 км
MIN_COUNTRY_FRACTION = 0.4             # частка квадрата сцени, що лежить в ESP/PRT (береги й кордони)
WAVE_OFFSETS = [(0.0, 0.0), (0.5, 0.5), (0.5, 0.0), (0.0, 0.5)]   # у частках кроку
N_WAVES = 1                            # 1 хвиля ~ все покриття без перекриття; 2+ -> зсунуті сітки з перекриттям (тоді CV лише блоками 2°)
YEARS = list(range(2017, 2026))
EARLIEST = {2017: date(2017, 7, 1)}    # Sentinel-2B запрацював улітку 2017
MAIN_SEASON = ((6, 1), (9, 30))
SHOULDER = [((3, 1), (5, 31)), ((10, 1), (10, 31))]
SHOULDER_FRAC = 0.25                   # 0.0 -> лише червень–вересень
SEED = 42


# ------------------------------------------------------------------ територія
def load_country_geometry():
    import geopandas as gpd
    import requests
    import shapely
    from shapely.geometry import box

    cache = RAW_DATA_DIR / "boundaries"
    cache.mkdir(parents=True, exist_ok=True)
    geoms = []
    for code in ("ESP", "PRT"):
        p = cache / f"{code}.geo.json"
        if not p.exists():
            url = f"https://raw.githubusercontent.com/johan/world.geo.json/master/countries/{code}.geo.json"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            p.write_text(r.text, encoding="utf-8")
        g = gpd.read_file(p)
        geoms.append(g.union_all() if hasattr(g, "union_all") else g.unary_union)
    geom = shapely.union_all(geoms)
    min_lat, min_lon, max_lat, max_lon = IBERIA
    geom = geom.intersection(box(min_lon, min_lat, max_lon, max_lat))
    shapely.prepare(geom)
    return geom


def country_fraction(inside_xy, lats, lons, tile_km=SCENE_KM, grid=7):
    """Частка квадрата сцени (вибірка grid x grid), що лежить у країні, + чи центр у країні."""
    offs = np.linspace(-tile_km / 2, tile_km / 2, grid)
    dy, dx = np.meshgrid(offs, offs, indexing="ij")
    dy, dx = dy.ravel(), dx.ravel()
    s_lat = lats[:, None] + dy[None, :] / 111.0
    s_lon = lons[:, None] + dx[None, :] / (111.0 * np.cos(np.radians(lats))[:, None])
    inside = np.asarray(inside_xy(s_lon.ravel(), s_lat.ravel())).reshape(s_lat.shape)
    return inside.mean(axis=1), np.asarray(inside_xy(lons, lats))


def grid_candidates(rng, fx, fy):
    min_lat, min_lon, max_lat, max_lon = IBERIA
    rows = []
    for lat in np.arange(min_lat + fy * STEP_DEG, max_lat, STEP_DEG):
        step_lon = STEP_DEG / np.cos(np.radians(lat))
        for lon in np.arange(min_lon + fx * step_lon, max_lon, step_lon):
            rows.append((lat + rng.uniform(-JITTER_DEG, JITTER_DEG),
                         lon + rng.uniform(-JITTER_DEG, JITTER_DEG) / np.cos(np.radians(lat))))
    return np.array(rows)


def coverage_fraction(inside_xy, centers, n=30000, half_km=SCENE_KM / 2, seed=0):
    """Частка території країни, що потрапляє хоча б в одну сцену (квадрат ±half_km від центру)."""
    rng = np.random.default_rng(seed)
    min_lat, min_lon, max_lat, max_lon = IBERIA
    plat = rng.uniform(min_lat, max_lat, n * 4)
    plon = rng.uniform(min_lon, max_lon, n * 4)
    ok = np.asarray(inside_xy(plon, plat))
    plat, plon = plat[ok][:n], plon[ok][:n]
    covered = np.zeros(len(plat), dtype=bool)
    for s in range(0, len(plat), 4000):
        la, lo = plat[s:s + 4000, None], plon[s:s + 4000, None]
        dlat = np.abs(la - centers[None, :, 0]) * 111.0 <= half_km
        dlon = np.abs(lo - centers[None, :, 1]) * 111.0 * np.cos(np.radians(la)) <= half_km
        covered[s:s + 4000] = (dlat & dlon).any(axis=1)
    return float(covered.mean()), len(plat)


# ------------------------------------------------------------------ дати
def balanced_years(rng, n, years):
    ys = np.resize(np.array(years), n)
    rng.shuffle(ys)
    return ys


def random_day(rng, year, shoulder_frac):
    windows = [MAIN_SEASON] if rng.random() >= shoulder_frac else SHOULDER
    earliest = EARLIEST.get(year)
    spans = []
    for (m0, d0), (m1, d1) in windows:
        a, b = date(year, m0, d0), date(year, m1, d1)
        if earliest:
            a = max(a, earliest)
        if a <= b:
            spans.append((a, b))
    if not spans:
        spans = [(max(date(year, 6, 1), earliest or date(year, 6, 1)), date(year, 9, 30))]
    lens = [(b - a).days + 1 for a, b in spans]
    k = int(rng.integers(0, sum(lens)))                 # рівномірно по днях усіх вікон
    for (a, _), L in zip(spans, lens):
        if k < L:
            return (a + timedelta(days=k)).isoformat()
        k -= L


def build_dates(rng, n, years=YEARS, shoulder_frac=SHOULDER_FRAC):
    return [random_day(rng, int(y), shoulder_frac) for y in balanced_years(rng, n, years)]


# ------------------------------------------------------------------ main
def main():
    import shapely

    rng = np.random.default_rng(SEED)
    geom = load_country_geometry()
    inside_xy = lambda x, y: shapely.contains_xy(geom, x, y)

    waves = []
    for w in range(N_WAVES):
        fx, fy = WAVE_OFFSETS[w]
        pts = grid_candidates(rng, fx, fy)
        frac, center = country_fraction(inside_xy, pts[:, 0], pts[:, 1])
        pts = pts[center & (frac >= MIN_COUNTRY_FRACTION)]
        waves.append(pts)
        cov, n_s = coverage_fraction(inside_xy, pts)
        print(f"хвиля {w + 1}: {len(pts)} локацій; покриття території ESP+PRT: {cov:.1%} (за {n_s} випадковими точками)")
    allc = np.vstack(waves)
    cov, _ = coverage_fraction(inside_xy, allc)
    print(f"усі хвилі разом: {len(allc)} локацій; покриття: {cov:.1%}")

    recs = []
    for w, pts in enumerate(waves, start=1):
        dates = build_dates(rng, len(pts))
        for i, (la, lo) in enumerate(pts):
            recs.append({"lat": round(float(la), 5), "lon": round(float(lo), 5), "target_date": dates[i],
                         "is_fire": 0, "loc_id": w * 100000 + i, "wave": w})
    df = pd.DataFrame(recs)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for w in range(1, N_WAVES + 1):
        df[df["wave"] == w].sample(frac=1.0, random_state=SEED + w).reset_index(drop=True) \
            .to_csv(INTERIM_DATA_DIR / f"points_grid_100m_wave{w}.csv", index=False)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.to_csv(INTERIM_DATA_DIR / "points_grid_100m.csv", index=False)

    d = pd.to_datetime(df["target_date"])
    print(f"\nсцен-кандидатів: {len(df)}")
    print("за роками:", d.dt.year.value_counts().sort_index().to_dict())
    print("за місяцями:", d.dt.month.value_counts().sort_index().to_dict())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))
        for w in range(1, N_WAVES + 1):
            s = df[df["wave"] == w]
            ax.scatter(s["lon"], s["lat"], s=6, label=f"хвиля {w} ({len(s)})")
        ax.set_aspect(1 / np.cos(np.radians(40)))
        ax.legend(); ax.set_title("Кандидати сцен 12.8 км (ESP+PRT)")
        fig.savefig(INTERIM_DATA_DIR / "points_grid_100m_map.png", dpi=130, bbox_inches="tight")
        print("карта:", INTERIM_DATA_DIR / "points_grid_100m_map.png")
    except Exception as e:
        print("карту не збережено:", e)


if __name__ == "__main__":
    main()