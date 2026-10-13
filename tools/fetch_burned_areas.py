import time
from pathlib import Path
import requests
import geopandas as gpd
import pandas as pd

from src.config import BASE_DIR

# directory where fire_date.py will look for these polygons
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "globfire"

# bounding box for iberia: min_lon, min_lat, max_lon, max_lat
IBERIA_BBOX = (-9.5, 35.9, 3.4, 43.9)
YEARS = [2020, 2021, 2022, 2023, 2024]


def fetch_effis_burned_areas(year: int, bbox: tuple, max_retries: int = 3) -> gpd.GeoDataFrame | None:
    """fetch burned area polygons from open wfs api copernicus effis with layer fallback"""
    min_lon, min_lat, max_lon, max_lat = bbox
    wfs_url = "https://ies-ows.jrc.ec.europa.eu/effis"
    
    # Possible layer naming conventions in EFFIS WFS
    possible_layers = [
        f"effis:burntareas_viirs_{year}",
        f"effis:viirs.ba.poly.{year}",
        f"effis:modis.ba.poly.{year}",
        f"effis:modis.ba.poly_{year}",
        f"effis:burntareas_{year}"
    ]
    
    for layer_name in possible_layers:
        params = {
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typeName": layer_name,
            "outputFormat": "application/json",
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat},EPSG:4326"
        }
        
        for attempt in range(max_retries):
            try:
                print(f"  [{year}] trying layer: {layer_name} (attempt {attempt + 1})...")
                response = requests.get(wfs_url, params=params, timeout=90)
                
                # If HTTP error (e.g. 404), break and try next layer
                if response.status_code != 200:
                    break 
                
                try:
                    data = response.json()
                except ValueError:
                    # Server returned XML (ExceptionReport) instead of JSON
                    error_text = response.text[:200].replace('\n', ' ')
                    print(f"    -> xml error (layer likely not found): {error_text}")
                    break # Move to the next layer in the list
                
                features = data.get("features", [])
                if not features:
                    print(f"    -> no fires found in this bbox for {layer_name}")
                    return None
                    
                # Load JSON into spatial dataframe
                gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                
                # Standardize date columns for fire_date.py
                cols_lower = {c.lower(): c for c in gdf.columns}
                start_col = cols_lower.get('firedate') or cols_lower.get('initialdate')
                end_col = cols_lower.get('lastupdate') or cols_lower.get('finaldate')
                
                if start_col and end_col:
                    gdf['InitialDate'] = pd.to_datetime(gdf[start_col]).dt.strftime('%Y-%m-%d')
                    gdf['FinalDate'] = pd.to_datetime(gdf[end_col]).dt.strftime('%Y-%m-%d')
                else:
                    print(f"    -> warning: could not find start/end dates")
                    
                return gdf

            except Exception as e:
                print(f"    -> network error: {e}")
                time.sleep(5)
                
    return None


def build_local_database():
    """check cache and download missing polygons"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_gdfs = []
    for year in YEARS:
        out_file = OUTPUT_DIR / f"effis_burnt_areas_viirs_{year}.geojson"
        
        if out_file.exists():
            print(f"[{year}] file already exists: {out_file.name} (skipping download)")
            try:
                gdf = gpd.read_file(out_file)
                all_gdfs.append(gdf)
            except Exception as e:
                print(f"  error reading cache: {e}")
            continue
            
        print(f"fetching fire polygons for {year}...")
        gdf = fetch_effis_burned_areas(year, IBERIA_BBOX)
        
        if gdf is not None and not gdf.empty:
            gdf.to_file(out_file, driver="GeoJSON")
            print(f"  [{year}] successfully saved {len(gdf)} polygons -> {out_file.name}")
            all_gdfs.append(gdf)
        else:
            print(f"  [{year}] skipped (no data found across all layer variants)")
            
    if all_gdfs:
        combined = pd.concat(all_gdfs, ignore_index=True)
        print(f"\ndone. your offline database contains {len(combined)} confirmed forest fires.")
    else:
        print("\nfailed to build burned areas database.")


if __name__ == "__main__":
    print("initializing local vector polygon database (geojson)...")
    build_local_database()