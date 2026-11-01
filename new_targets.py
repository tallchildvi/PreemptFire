"""Запусти ЛОКАЛЬНО (там, де лежить data/raw/globfire): вирізає Іспанію+Португалію з усіх річних файлів
вигорілих площ і пише один малий GeoPackage для завантаження в Kaggle як Dataset.

    python clip_globfire_iberia.py --src data/raw/globfire --out ba_iberia.gpkg
"""
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/raw/globfire")
ap.add_argument("--out", default="ba_iberia.gpkg")
ap.add_argument("--bbox", type=float, nargs=4, default=[-10.8, 35.5, 4.6, 44.3], metavar=("W", "S", "E", "N"))
a = ap.parse_args()

W, S, E, N = a.bbox
files = sorted(p for p in Path(a.src).rglob("*") if p.suffix.lower() in (".gpkg", ".geojson", ".shp"))
assert files, f"У {a.src} немає векторних файлів"
parts = []
for f in files:
    g = gpd.read_file(f)
    if g.empty:
        continue
    g = g.to_crs("EPSG:4326").cx[W:E, S:N].copy()
    g["src_file"] = f.name
    print(f"{f.name}: {len(g)} полігонів у bbox")
    if len(g):
        parts.append(g)
out = pd.concat(parts, ignore_index=True)
gpd.GeoDataFrame(out, crs="EPSG:4326").to_file(a.out, driver="GPKG")
print(f"Записано {len(out)} полігонів -> {a.out}")
print("Колонки:", [c for c in out.columns if c != "geometry"])