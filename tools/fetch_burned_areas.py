import argparse
import os
import re
import tempfile
import time
from pathlib import Path
import requests
import geopandas as gpd
import pandas as pd
from shapely.ops import transform

from src.config import BASE_DIR


def _ensure_lon_lat_order(gdf: gpd.GeoDataFrame, bbox: tuple) -> gpd.GeoDataFrame:
    """fix mapserver geojson bug where coordinates are returned as [lat, lon] instead of [lon, lat]"""
    if gdf.empty:
        return gdf

    min_lon, min_lat, max_lon, max_lat = bbox
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # check if x/y fit better when swapped (mapserver wfs 1.0.0 geojson quirk)
    normal_ok = (min_lon - 5 <= bounds[0] <= max_lon + 5) and (min_lat - 5 <= bounds[1] <= max_lat + 5)
    swapped_ok = (min_lat - 5 <= bounds[0] <= max_lat + 5) and (min_lon - 5 <= bounds[1] <= max_lon + 5)

    if swapped_ok and not normal_ok:
        print("    -> detected inverted [lat, lon] axes from mapserver, swapping to [lon, lat]...")
        gdf = gdf.copy()
        gdf["geometry"] = gdf["geometry"].map(
            lambda geom: transform(lambda x, y, *z: (y, x), geom) if geom is not None else None
        )
    return gdf


def _extract_mapserver_error(raw_text: str) -> str:
    """strip html comments and tags to show the actual mapserver error message"""
    cleaned = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[:200] if cleaned else raw_text[:200]


def fetch_effis_burned_areas(year: int, bbox: tuple, max_retries: int = 3) -> gpd.GeoDataFrame | None:
    """fetch burned area polygons from copernicus effis wfs mapserver"""
    min_lon, min_lat, max_lon, max_lat = bbox
    wfs_url = "https://maps.effis.emergency.copernicus.eu/effis"

    # official effis mapserver layer names (annual archives use modis.ba.poly.<year>)
    possible_layers = [
        f"ms:modis.ba.poly.{year}",
        f"modis.ba.poly.{year}",
        f"ms:viirs.ba.poly.{year}",
        f"viirs.ba.poly.{year}",
        "ms:effis.nrt.ba.poly",
    ]

    # try geojson first, fallback to default gml if mapserver rejects geojson
    output_formats = ["geojson", None]

    for layer_name in possible_layers:
        for fmt in output_formats:
            params = {
                "service": "WFS",
                "version": "1.0.0",
                "request": "GetFeature",
                "typename": layer_name,
                # strictly 4 values for wfs 1.0.0 (5th epsg param crashes mapserver 7.6)
                "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            }
            if fmt:
                params["outputformat"] = fmt

            for attempt in range(max_retries):
                try:
                    fmt_label = fmt if fmt else "gml"
                    print(f"  [{year}] trying layer: {layer_name} [{fmt_label}] (attempt {attempt + 1})...")
                    response = requests.get(wfs_url, params=params, timeout=240)

                    if response.status_code != 200:
                        print(f"    -> http status {response.status_code}")
                        break

                    text_sample = response.text[:600]
                    upper_sample = text_sample.upper()

                    if "<HTML>" in upper_sample or "MAPSERVER MESSAGE" in upper_sample:
                        err_msg = _extract_mapserver_error(response.text)
                        print(f"    -> mapserver message: {err_msg}")
                        break

                    if "EXCEPTIONREPORT" in upper_sample or "SERVICEEXCEPTION" in upper_sample:
                        err_msg = _extract_mapserver_error(response.text)
                        print(f"    -> wfs exception: {err_msg}")
                        break

                    # parse response depending on format
                    if fmt == "geojson":
                        data = response.json()
                        features = data.get("features", [])
                        if not features:
                            print(f"    -> 0 polygons returned in bbox for {layer_name}")
                            return None
                        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                    else:
                        with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
                            tmp.write(response.content)
                            tmp_path = tmp.name
                        try:
                            gdf = gpd.read_file(tmp_path)
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                    if gdf.empty:
                        print(f"    -> 0 polygons returned in bbox for {layer_name}")
                        return None

                    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
                        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

                    gdf = _ensure_lon_lat_order(gdf, bbox)

                    # standardize date columns (effis uses FIREDATE and FINALDATE / LASTUPDATE)
                    cols_lower = {c.lower(): c for c in gdf.columns}
                    start_col = (
                        cols_lower.get("firedate")
                        or cols_lower.get("initialdate")
                        or cols_lower.get("initdate")
                    )
                    end_col = (
                        cols_lower.get("finaldate")
                        or cols_lower.get("lastupdate")
                        or cols_lower.get("lastdate")
                    )

                    if start_col and end_col:
                        gdf["InitialDate"] = pd.to_datetime(gdf[start_col], errors="coerce").dt.strftime("%Y-%m-%d")
                        gdf["FinalDate"] = pd.to_datetime(gdf[end_col], errors="coerce").dt.strftime("%Y-%m-%d")
                        # filter by target year in case nrt fallback layer was used
                        gdf = gdf[gdf["InitialDate"].str.startswith(str(year), na=False)].copy()
                    else:
                        print(f"    -> warning: date columns not found in {list(gdf.columns)}")

                    if gdf.empty:
                        return None

                    return gdf

                except Exception as e:
                    print(f"    -> network/parsing error: {e}")
                    time.sleep(5)

    return None


def build_local_database(bbox: tuple, years: list, output_dir: Path):
    """check cache and download missing polygons for the given bounding box"""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_gdfs = []
    for year in years:
        out_file = output_dir / f"effis_burnt_areas_{year}.geojson"

        if out_file.exists():
            print(f"[{year}] file already exists: {out_file.name} (skipping download)")
            try:
                gdf = gpd.read_file(out_file)
                all_gdfs.append(gdf)
            except Exception as e:
                print(f"  error reading cache: {e}")
            continue

        print(f"fetching fire polygons for {year}...")
        gdf = fetch_effis_burned_areas(year, bbox)

        if gdf is not None and not gdf.empty:
            gdf.to_file(out_file, driver="GeoJSON")
            print(f"  [{year}] successfully saved {len(gdf)} polygons -> {out_file.name}")
            all_gdfs.append(gdf)
        else:
            print(f"  [{year}] skipped (no data found across all layer variants)")

    if all_gdfs:
        combined = pd.concat(all_gdfs, ignore_index=True)
        print(f"\ndone. offline database contains {len(combined)} confirmed forest fires.")
    else:
        print("\nfailed to build burned areas database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fetch burned area polygons from effis wfs")

    parser.add_argument("--min_lon", type=float, default=-9.5, help="minimum longitude (west)")
    parser.add_argument("--min_lat", type=float, default=35.9, help="minimum latitude (south)")
    parser.add_argument("--max_lon", type=float, default=3.4, help="maximum longitude (east)")
    parser.add_argument("--max_lat", type=float, default=43.9, help="maximum latitude (north)")

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024],
        help="list of years to fetch (e.g. 2020 2021)",
    )

    args = parser.parse_args()

    target_bbox = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    target_dir = BASE_DIR / "data" / "raw" / "globfire"

    print("initializing local vector polygon database (geojson)...")
    print(f"target bounding box: {target_bbox}")
    print(f"target years: {args.years}")

    build_local_database(bbox=target_bbox, years=args.years, output_dir=target_dir)