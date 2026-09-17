import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Callable, Dict, List, Optional, Tuple
import fiona
import geopandas as gpd
import osmnx as ox
import pandas as pd
from pyproj import Transformer


OSM_CACHE_DIR = Path("data/osm_cache")
OSM_TILES_DIR = OSM_CACHE_DIR / "tiles"
OSM_DB_PATH = OSM_CACHE_DIR / "osm_cache_index.db"

BBOX_SNAP_DEG = 0.5  # ~55 km grid snapping cell

OVERPASS_SERVERS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

OSM_FILTER_TAGS: Dict[str, List[str]] = {
    "highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service", "track", "path", "footway",
        "motorway_link", "trunk_link", "primary_link",
        "secondary_link", "tertiary_link",
    ],
    "railway": ["rail", "narrow_gauge", "spur"],
    "power": ["line", "minor_line", "cable", "substation"],
    "natural": ["water", "wetland"],
    "waterway": ["river", "stream", "canal", "riverbank"],
    "tourism": ["camp_site", "picnic_site", "wilderness_hut"],
    "amenity": ["shelter", "firepit"],
    "bridge": ["yes"],
}

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

def snap_bbox(
    west: float, south: float, east: float, north: float, snap: float = BBOX_SNAP_DEG
) -> Tuple[float, float, float, float]:
    """snaps bbox corners outward to coarse grid so adjacent queries share a cached tile."""
    return (
        math.floor(west / snap) * snap,
        math.floor(south / snap) * snap,
        math.ceil(east / snap) * snap,
        math.ceil(north / snap) * snap,
    )


def compute_cache_key(west: float, south: float, east: float, north: float, date_str: str) -> str:
    """computes deterministic 16-character sha1 key from snapped bbox and date."""
    snapped = snap_bbox(west, south, east, north)
    raw_payload = json.dumps([*snapped, date_str], sort_keys=True)
    return hashlib.sha1(raw_payload.encode("utf-8")).hexdigest()[:16]


def get_gpkg_path(key: str) -> Path:
    return OSM_TILES_DIR / f"{key}.gpkg"

def init_cache_db() -> sqlite3.Connection:
    """initializes sqlite metadata index with wal mode for concurrent safety."""
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OSM_TILES_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(OSM_DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS osm_cache (
            cache_key     TEXT PRIMARY KEY,
            west          REAL NOT NULL,
            south         REAL NOT NULL,
            east          REAL NOT NULL,
            north         REAL NOT NULL,
            date_str      TEXT NOT NULL,
            gpkg_path     TEXT NOT NULL,
            feature_count INTEGER NOT NULL,
            created_at    TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def index_get(conn: sqlite3.Connection, key: str) -> Optional[Tuple[str, int]]:
    """fetches cached record (gpkg_path, feature_count) if valid."""
    cursor = conn.cursor()
    cursor.execute("SELECT gpkg_path, feature_count FROM osm_cache WHERE cache_key = ?", (key,))
    row = cursor.fetchone()
    if row:
        path_str, feature_count = row
        # if 0 features, no file is required on disk
        if feature_count == 0 or Path(path_str).exists():
            return path_str, feature_count
        # remove stale entry if file was deleted
        cursor.execute("DELETE FROM osm_cache WHERE cache_key = ?", (key,))
        conn.commit()
    return None


def index_put(
    conn: sqlite3.Connection,
    key: str,
    west: float,
    south: float,
    east: float,
    north: float,
    date_str: str,
    gpkg_path: Path,
    feature_count: int,
) -> None:
    """records newly cached tile in sqlite index."""
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO osm_cache 
            (cache_key, west, south, east, north, date_str, gpkg_path, feature_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key, west, south, east, north, date_str, str(gpkg_path), feature_count, now_str),
        )

def fetch_overpass_at_date(
    west: float, south: float, east: float, north: float, date_str: str
) -> Optional[gpd.GeoDataFrame]:
    """queries overpass with exact historical timestamp and multi-mirror fallback."""

    ox.settings.overpass_settings = f'[out:json][timeout:90][date:"{date_str}T00:00:00Z"]'
    ox.settings.use_cache = False
    ox.settings.log_console = False
    ox.settings.http_user_agent = "WildfireResearchPipeline/1.0 (academic wildfire risk modeling)"
    ox.settings.requests_kwargs = {}
        
    for endpoint in OVERPASS_SERVERS:
        ox.settings.overpass_endpoint = endpoint

        for attempt in range(MAX_RETRIES):
            try:
                if hasattr(ox, "features_from_bbox"):
                    gdf = ox.features_from_bbox(
                        bbox=(west, south, east, north),
                        tags=OSM_FILTER_TAGS,
                    )
                else:
                    gdf = ox.geometries_from_bbox(
                        north=north, south=south, east=east, west=west,
                        tags=OSM_FILTER_TAGS,
                    )

                if gdf is not None and not gdf.empty:
                    return gdf

                # valid response for remote wilderness (0 features)
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            except Exception as e:
                wait_time = RETRY_BACKOFF ** attempt
                print(f"  [OSM] {endpoint} attempt {attempt + 1}/{MAX_RETRIES} failed ({e}), retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)

        print(f"  [OSM] {endpoint} exhausted — trying next mirror...")

    print("  [OSM] all mirrors failed for this bbox query.")
    return None

_db_connection: Optional[sqlite3.Connection] = None


def get_db_connection() -> sqlite3.Connection:
    global _db_connection
    if _db_connection is None:
        _db_connection = init_cache_db()
    return _db_connection


def query_osm_at_date(
    west: float,
    south: float,
    east: float,
    north: float,
    target_date: str,
    target_crs: str = "EPSG:4326",
) -> Optional[gpd.GeoDataFrame]:
    """retrieves osm features from local disk cache or falls back to overpass."""
    snapped = snap_bbox(west, south, east, north)
    cache_key = compute_cache_key(*snapped, target_date)
    conn = get_db_connection()

    # 1. cache hit check
    cached_record = index_get(conn, cache_key)
    if cached_record:
        gpkg_path_str, feature_count = cached_record
        if feature_count == 0:
            return gpd.GeoDataFrame(geometry=[], crs=target_crs)

        try:
            layers = fiona.listlayers(gpkg_path_str)
            gdfs = []
            for layer in layers:
                gdf_layer = gpd.read_file(
                    gpkg_path_str,
                    layer=layer,
                    bbox=(west, south, east, north),
                )
                if not gdf_layer.empty:
                    gdfs.append(gdf_layer)

            if not gdfs:
                return gpd.GeoDataFrame(geometry=[], crs=target_crs)

            merged = pd.concat(gdfs, ignore_index=True)
            if merged.crs is None:
                merged = merged.set_crs("EPSG:4326")
            return merged.to_crs(target_crs) if str(merged.crs) != target_crs else merged

        except Exception as e:
            print(f"  [OSM Cache] corrupt shard ({e}) — refreshing from overpass...")

    # 2. cache miss: fetch from overpass
    print(f"  [OSM] cache miss for {target_date} bbox=[{snapped[0]:.1f},{snapped[1]:.1f},{snapped[2]:.1f},{snapped[3]:.1f}]")
    gdf = fetch_overpass_at_date(*snapped, date_str=target_date)
    if gdf is None:
        return None

    gpkg_file = get_gpkg_path(cache_key)
    feature_count = len(gdf)

    # 3. persist valid features to geopackage
    if feature_count > 0:
        valid_cols = [
            c for c in gdf.columns
            if c in {"geometry", "name", "bridge", *OSM_FILTER_TAGS.keys()}
        ]
        gdf[valid_cols].to_file(gpkg_file, layer="osm_features", driver="GPKG")

    index_put(conn, cache_key, *snapped, target_date, gpkg_file, feature_count)
    print(f"  [OSM] cached {feature_count} features → {gpkg_file.name}")

    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return gdf.to_crs(target_crs) if str(gdf.crs) != target_crs else gdf


def make_temporal_osm_query() -> Callable:
    """creates callable shim for SpatialFeatureFetcher integration."""
    def _query(grid_info: dict, target_date: str, buffer_meters: float = 4000.0) -> Optional[gpd.GeoDataFrame]:
        min_x, min_y, max_x, max_y = grid_info["utm_bounds"]
        transformer = Transformer.from_crs(grid_info["crs"], "EPSG:4326", always_xy=True)

        corners = [
            (min_x - buffer_meters, min_y - buffer_meters),
            (min_x - buffer_meters, max_y + buffer_meters),
            (max_x + buffer_meters, min_y - buffer_meters),
            (max_x + buffer_meters, max_y + buffer_meters),
        ]
        wgs84_coords = [transformer.transform(x, y) for x, y in corners]
        west = min(p[0] for p in wgs84_coords)
        south = min(p[1] for p in wgs84_coords)
        east = max(p[0] for p in wgs84_coords)
        north = max(p[1] for p in wgs84_coords)

        return query_osm_at_date(
            west=west, south=south, east=east, north=north,
            target_date=target_date,
            target_crs=grid_info["crs"],
        )

    return _query


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pre-warm OSM temporal cache from points CSV")
    parser.add_argument("--csv", required=True, help="path to points CSV (lat, lon, acq_date)")
    parser.add_argument("--buffer", default=0.25, type=float, help="search radius buffer in degrees")
    args = parser.parse_args()

    df_points = pd.read_csv(args.csv)
    date_col = "acq_date" if "acq_date" in df_points.columns else "target_date"
    lat_col = "latitude" if "latitude" in df_points.columns else "lat"
    lon_col = "longitude" if "longitude" in df_points.columns else "lon"

    seen_keys = set()
    unique_requests = []

    for _, r in df_points.iterrows():
        lat_val, lon_val = float(r[lat_col]), float(r[lon_col])
        d_str = str(r[date_col]).split(" ")[0]
        bbox_snapped = snap_bbox(
            lon_val - args.buffer, lat_val - args.buffer,
            lon_val + args.buffer, lat_val + args.buffer
        )
        key_id = compute_cache_key(*bbox_snapped, d_str)
        if key_id not in seen_keys:
            seen_keys.add(key_id)
            unique_requests.append((bbox_snapped, d_str))

    print(f"starting cache pre-warming: {len(df_points)} points mapped to {len(unique_requests)} unique tiles...")

    for idx, (bbox_coords, d_val) in enumerate(unique_requests, 1):
        print(f"\n[{idx}/{len(unique_requests)}] caching {d_val} for bbox={bbox_coords}")
        query_osm_at_date(*bbox_coords, target_date=d_val, target_crs="EPSG:4326")

    print("\ncache pre-warming complete.")