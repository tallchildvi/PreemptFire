from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union
from dotenv import load_dotenv
from filelock import FileLock
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from pyrosm import OSM
import requests
import tempfile
from shapely.geometry import box

load_dotenv()

DEFAULT_CACHE_DIR = Path("data/osm_raw")
GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"

OSM_EXTRACTION_FILTER: Dict[str, Any] = {
    "highway": True,
    "waterway": True,
    "natural": ["water", "wetland"],
    "railway": ["rail", "narrow_gauge", "spur"],
    "power": ["line", "minor_line", "cable", "substation"],
    "tourism": ["camp_site", "picnic_site", "wilderness_hut"],
    "amenity": ["shelter", "firepit"],
    "bridge": ["yes"],
}


def parse_history_url(urls_payload: Any) -> Optional[str]:
    """Extracts internal history PBF download link from payload dictionary/string."""
    if isinstance(urls_payload, dict):
        return urls_payload.get("history")
    if isinstance(urls_payload, str):
        try:
            parsed = json.loads(urls_payload)
            return parsed.get("history") if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def load_geofabrik_index(cache_dir: Path) -> gpd.GeoDataFrame:
    """Fetches, parses and caches global Geofabrik spatial index."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "geofabrik_index.json"

    if not index_path.exists():
        print("  [OSM] Fetching global Geofabrik spatial index...")
        response = requests.get(GEOFABRIK_INDEX_URL, timeout=30)
        response.raise_for_status()
        index_path.write_text(response.text, encoding="utf-8")

    gdf = gpd.read_file(index_path)
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
    gdf["history_url"] = gdf["urls"].apply(parse_history_url)
    return gdf


def resolve_finest_covering_regions(
    index_gdf: gpd.GeoDataFrame,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    polygon_mask: Optional[Polygon | MultiPolygon] = None,
    allowed_countries: Optional[List[str]] = None,
    allowed_regions: Optional[List[str]] = None,
    exclude_regions: Optional[List[str]] = None,
    clip_to_allowed_countries: bool = False,
) -> List[Tuple[str, str]]:
    """resolves covering regions using adaptive polygons, boundary whitelists and blacklists."""
    
    # 1. build query spatial geometry (adaptive polygon or fallback to rectangular box)
    if polygon_mask is not None:
        query_geometry = polygon_mask
    elif bbox is not None:
        query_geometry = box(*bbox)
    else:
        raise ValueError("either bbox or polygon_mask must be provided.")

    # 2. build lookup table
    node_map: Dict[str, pd.Series] = {}
    for _, row in index_gdf.iterrows():
        node_id = row.get("id")
        geom = row.get("geometry")
        if pd.notna(node_id) and geom is not None and not geom.is_empty:
            node_map[str(node_id)] = row

    all_ids = set(node_map.keys())

    # 3. resolve hierarchical parents
    logical_parent: Dict[str, Optional[str]] = {}
    for node_id, row in node_map.items():
        parent_raw = row.get("parent")
        parent_id = str(parent_raw) if pd.notna(parent_raw) and str(parent_raw).strip() else None
        if "/" in node_id:
            path_parent = node_id.rsplit("/", 1)[0]
            if path_parent in node_map:
                logical_parent[node_id] = path_parent
                continue
        logical_parent[node_id] = parent_id

    children_map: Dict[str, List[str]] = {}
    for node_id, parent_id in logical_parent.items():
        if parent_id is not None and parent_id in all_ids:
            children_map.setdefault(parent_id, []).append(node_id)

    # 4. optional boundary clipping: intersect query geometry with real shapes of allowed countries
    if clip_to_allowed_countries and allowed_countries:
        c_upper = {c.upper().strip() for c in allowed_countries}
        c_clean = {c.lower().strip() for c in allowed_countries}
        matching_geoms = []
        for n_id, row in node_map.items():
            iso_raw = row.get("iso3166-1:alpha2")
            iso_list = [str(x).upper() for x in (iso_raw if isinstance(iso_raw, (list, tuple, np.ndarray, set)) else [iso_raw])] if iso_raw is not None else []
            if any(c in iso_list for c in c_upper) or n_id.lower() in c_clean:
                matching_geoms.append(row.geometry)
        if matching_geoms:
            target_country_union = unary_union(matching_geoms)
            query_geometry = query_geometry.intersection(target_country_union)

    # prepare blacklist lookup
    exclude_clean = {e.lower().strip() for e in (exclude_regions or [])}

    def has_valid_tag(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, (list, tuple, np.ndarray, set)):
            return len(val) > 0
        try:
            return bool(pd.notna(val) and str(val).strip())
        except (ValueError, TypeError):
            return False

    def is_jurisdiction_allowed(node_id: str) -> bool:
        # check blacklist first
        node_slug = node_id.lower().split("/")[-1]
        if node_id.lower() in exclude_clean or node_slug in exclude_clean:
            return False

        if not allowed_countries and not allowed_regions:
            return True

        row = node_map.get(node_id)
        if row is None:
            return False

        if allowed_regions:
            allowed_reg_clean = {r.lower().strip() for r in allowed_regions}
            if node_id.lower() in allowed_reg_clean or node_slug in allowed_reg_clean:
                return True

        if allowed_countries:
            allowed_c_upper = {c.upper().strip() for c in allowed_countries}
            allowed_c_clean = {c.lower().strip() for c in allowed_countries}

            curr: Optional[str] = node_id
            while curr is not None:
                curr_row = node_map.get(curr)
                if curr_row is not None:
                    iso_raw = curr_row.get("iso3166-1:alpha2")
                    if isinstance(iso_raw, (list, tuple, np.ndarray, set)):
                        if any(str(x).upper() in allowed_c_upper for x in iso_raw):
                            return True
                    elif has_valid_tag(iso_raw) and str(iso_raw).upper() in allowed_c_upper:
                        return True

                    iso2_raw = curr_row.get("iso3166-2")
                    if isinstance(iso2_raw, (list, tuple, np.ndarray, set)):
                        if any(str(x).split("-")[0].upper() in allowed_c_upper for x in iso2_raw if "-" in str(x)):
                            return True
                    elif has_valid_tag(iso2_raw) and "-" in str(iso2_raw):
                        if str(iso2_raw).split("-")[0].upper() in allowed_c_upper:
                            return True

                    if any(part in allowed_c_clean for part in curr.lower().split("/")):
                        return True
                curr = logical_parent.get(curr)

        return False

    def intersects_query(node_id: str) -> bool:
        row = node_map.get(node_id)
        if row is None or row.geometry is None or row.geometry.is_empty:
            return False
        try:
            # check intersection with meaningful spatial overlap
            return bool(row.geometry.intersects(query_geometry))
        except Exception:
            return False

    history_cache: Dict[str, bool] = {}

    def has_history(node_id: str, visited: Optional[Set[str]] = None) -> bool:
        if node_id in history_cache:
            return history_cache[node_id]
        if visited is None:
            visited = set()
        if node_id in visited:
            return False
        visited.add(node_id)

        row = node_map.get(node_id)
        if row is None:
            history_cache[node_id] = False
            return False

        h_url = row.get("history_url")
        if pd.notna(h_url) and str(h_url).strip():
            history_cache[node_id] = True
            return True

        for child_id in children_map.get(node_id, []):
            if has_history(child_id, visited.copy()):
                history_cache[node_id] = True
                return True

        history_cache[node_id] = False
        return False

    continents = {
        "north-america", "south-america", "europe", "asia", "africa", "oceania", "australia-oceania",
    }

    def is_special_region(node_id: str) -> bool:
        if node_id in continents:
            return False
        row = node_map.get(node_id)
        if row is None:
            return False
        parent_id = logical_parent.get(node_id)
        if parent_id not in continents or "/" in node_id:
            return False
        if has_valid_tag(row.get("iso3166-1:alpha2")) or has_valid_tag(row.get("iso3166-2")):
            return False
        return not bool(children_map.get(node_id))

    def collect_deepest(node_id: str, visited: Optional[Set[str]] = None) -> List[str]:
        if visited is None:
            visited = set()
        if node_id in visited:
            return []
        visited = visited.copy()
        visited.add(node_id)

        if not intersects_query(node_id) or not is_jurisdiction_allowed(node_id):
            return []

        row = node_map.get(node_id)
        if row is None:
            return []

        valid_children = [
            cid for cid in children_map.get(node_id, [])
            if cid != node_id
            and not is_special_region(cid)
            and intersects_query(cid)
            and is_jurisdiction_allowed(cid)
            and has_history(cid)
        ]

        if valid_children:
            res = []
            for cid in valid_children:
                res.extend(collect_deepest(cid, visited))
            if res:
                return res

        h_url = row.get("history_url")
        if pd.notna(h_url) and str(h_url).strip():
            return [node_id]
        return []

    # root traversal
    root_nodes = [
        nid for nid, row in node_map.items()
        if (logical_parent.get(nid) is None or logical_parent.get(nid) not in all_ids)
        and not is_special_region(nid)
        and intersects_query(nid)
        and is_jurisdiction_allowed(nid)
        and has_history(nid)
    ]

    resolved_ids: List[str] = []
    seen: Set[str] = set()
    for root_id in root_nodes:
        for reg_id in collect_deepest(root_id):
            if reg_id not in seen:
                seen.add(reg_id)
                resolved_ids.append(reg_id)

    output: List[Tuple[str, str]] = []
    for reg_id in resolved_ids:
        row = node_map.get(reg_id)
        if row is not None and pd.notna(row.get("history_url")):
            output.append((reg_id, str(row["history_url"])))

    return output

def plot_bbox_coverage(
    index_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    resolved_regions: List[Tuple[str, str]],
    output_html: Path | str = "data/osm_raw/coverage_map.html",
) -> Path:
    """Generates an interactive HTML map showing the BBox and resolved covering OSM regions."""
    west, south, east, north = bbox
    query_box = box(west, south, east, north)

    bbox_gdf = gpd.GeoDataFrame(
        [{"name": "Target Scene / Macro BBox", "geometry": query_box}],
        crs="EPSG:4326",
    )

    resolved_ids = {r[0] for r in resolved_regions}
    selected_gdf = index_gdf[index_gdf["id"].isin(resolved_ids)].copy()

    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri)",
    ).add_to(m)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, (_, row) in enumerate(selected_gdf.iterrows()):
        color = colors[idx % len(colors)]
        reg_id = str(row["id"])
        h_url = str(row.get("history_url", ""))

        geo_json = folium.GeoJson(
            row.geometry,
            name=f"Region: {reg_id}",
            style_function=lambda x, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.25,
            },
            highlight_function=lambda x: {
                "weight": 4,
                "fillOpacity": 0.5,
            },
            tooltip=folium.Tooltip(
                f"<b>ID:</b> {reg_id}<br><b>URL:</b> {h_url}",
                sticky=True,
            ),
        )
        geo_json.add_to(m)

    folium.GeoJson(
        bbox_gdf,
        name="Target Bounding Box",
        style_function=lambda x: {
            "fillColor": "red",
            "color": "red",
            "weight": 3,
            "dashArray": "6, 6",
            "fillOpacity": 0.08,
        },
        tooltip=folium.Tooltip(
            f"<b>Target BBox:</b> [{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]",
            sticky=True,
        ),
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"  [Map] Interactive coverage map saved to: {out_path.resolve()}")
    return out_path

def ensure_history_dump(region_id: str, history_url: str, cache_dir: Path) -> Path:
    """Downloads authenticated historical .osh.pbf file with binary validation."""
    clean_id = region_id.replace("/", "_")
    target_path = cache_dir / f"{clean_id}-history.osh.pbf"
    lock_path = target_path.with_suffix(".lock")

    if target_path.exists() and target_path.stat().st_size > 10_485_760:
        return target_path

    cookie = os.getenv("GEOFABRIK_COOKIE")
    if not cookie:
        raise ValueError("GEOFABRIK_COOKIE environment variable is missing in .env")

    cache_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=7200):
        if target_path.exists() and target_path.stat().st_size > 10_485_760:
            return target_path

        print(f"\n  [OSM] Downloading historical dump '{region_id}'\n        {history_url}")
        tmp_path = target_path.with_suffix(".tmp")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Cookie": cookie.strip('"\''),
        }

        with requests.get(history_url, headers=headers, stream=True, timeout=60, allow_redirects=True) as resp:
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise PermissionError(
                    f"Authentication failed for {history_url}. Server returned an HTML login page. "
                    "Your GEOFABRIK_COOKIE has expired. Please refresh it in your .env file."
                )

            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            t0 = time.perf_counter()

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = downloaded * 100.0 / total_size
                            mb_d = downloaded / 1_048_576
                            mb_t = total_size / 1_048_576
                            speed = mb_d / max(0.1, time.perf_counter() - t0)
                            sys.stdout.write(f"\r  [Download] {mb_d:.1f}/{mb_t:.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s")
                            sys.stdout.flush()

            if tmp_path.stat().st_size < 10_485_760:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded file {target_path.name} is smaller than 10MB (corrupted).")

            tmp_path.rename(target_path)
            print(f"\n  [OSM] Saved to {target_path.name} ({target_path.stat().st_size / (1024 * 1024):.1f} MB)")

    return target_path


def get_or_create_snapshot_pbf(osh_path: Path, target_date_str: str) -> Path:
    """Generates provincial point-in-time snapshot via C++ Osmium (cached per date)."""
    date_tag = target_date_str.split(" ")[0].replace("-", "")
    snapshot_path = osh_path.parent / f"{osh_path.stem}_{date_tag}.osm.pbf"
    lock_path = snapshot_path.with_suffix(".lock")

    if snapshot_path.exists():
        return snapshot_path

    with FileLock(str(lock_path), timeout=1800):
        if snapshot_path.exists():
            return snapshot_path

        iso_timestamp = f"{target_date_str.split(' ')[0]}T23:59:59Z"
        print(f"  [OSM Time-Filter] Generating snapshot for {iso_timestamp}")
        t0 = time.perf_counter()

        cmd = [
            "osmium", "time-filter",
            str(osh_path),
            iso_timestamp,
            "-o", str(snapshot_path),
            "--overwrite"
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"osmium time-filter failed: {res.stderr}")

        print(f"  [OSM Time-Filter] Snapshot created: {snapshot_path.name} in {time.perf_counter() - t0:.2f}s")

    return snapshot_path


def extract_scene_bbox_pbf(
    source_pbf: Path,
    bbox: Tuple[float, float, float, float],
    cache_dir: Path,
) -> Path:
    """Crops exact scene BBox from provincial snapshot using C++ Osmium in ~0.3s."""
    west, south, east, north = bbox
    bbox_tag = f"{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}".replace("-", "m").replace(".", "p")
    scene_pbf_path = cache_dir / f"scene_{source_pbf.stem}_{bbox_tag}.osm.pbf"

    if scene_pbf_path.exists():
        return scene_pbf_path

    cmd = [
        "osmium", "extract",
        "--strategy", "smart",
        "--bbox", f"{west},{south},{east},{north}",
        str(source_pbf),
        "-o", str(scene_pbf_path),
        "--overwrite"
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"osmium extract failed: {res.stderr}")

    return scene_pbf_path


class HistoricalOSMExtractor:
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        allowed_countries: Optional[List[str]] = None,
        allowed_regions: Optional[List[str]] = None,
        exclude_regions: Optional[List[str]] = None,
    ):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_gdf = load_geofabrik_index(self.cache_dir)
        self.allowed_countries = allowed_countries
        self.allowed_regions = allowed_regions
        self.exclude_regions = exclude_regions

    def resolve_history_files(
        self, west: float, south: float, east: float, north: float
    ) -> List[Dict[str, Any]]:
        """Resolves finest covering regions for a BBox taking filters into account."""
        regions = resolve_finest_covering_regions(
            self.index_gdf,
            bbox=(west, south, east, north),
            allowed_countries=self.allowed_countries,
            allowed_regions=self.allowed_regions,
            exclude_regions=self.exclude_regions,
        )
        output = []

        for region_id, history_url in regions:
            clean_id = region_id.replace("/", "_")
            local_path = self.cache_dir / f"{clean_id}-history.osh.pbf"
            is_valid = local_path.exists() and local_path.stat().st_size > 10_485_760
            output.append({
                "region_id": region_id,
                "history_url": history_url,
                "local_path": local_path,
                "downloaded": is_valid,
            })

        return output
    def download_required_dumps(self, resolved_files: List[Dict[str, Any]]) -> List[Path]:
        """downloads all missing regional .osh.pbf files."""
        return [
            ensure_history_dump(entry["region_id"], entry["history_url"], self.cache_dir)
            for entry in resolved_files
        ]

    def extract_features_for_date(
        self,
        date: str | datetime,
        west: float,
        south: float,
        east: float,
        north: float,
        target_crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        date_str = (
            date.strftime("%Y-%m-%d")
            if isinstance(date, datetime)
            else date.split(" ")[0]
        )
        resolved_files = self.resolve_history_files(west, south, east, north)
        gathered_gdfs: List[gpd.GeoDataFrame] = []

        for entry in resolved_files:
            osh_path = ensure_history_dump(
                entry["region_id"], entry["history_url"], self.cache_dir
            )
            snapshot_pbf = get_or_create_snapshot_pbf(osh_path, date_str)

            with tempfile.NamedTemporaryFile(
                suffix=".osm.pbf", dir=self.cache_dir, delete=False
            ) as tmp:
                tmp_scene_path = Path(tmp.name)

            try:
                cmd = [
                    "osmium",
                    "extract",
                    "--strategy",
                    "smart",
                    "--bbox",
                    f"{west},{south},{east},{north}",
                    str(snapshot_pbf),
                    "-o",
                    str(tmp_scene_path),
                    "--overwrite",
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"osmium extract failed: {res.stderr}")

                osm = OSM(str(tmp_scene_path))
                gdf_chunk = osm.get_data_by_custom_criteria(
                    custom_filter=OSM_EXTRACTION_FILTER,
                    filter_type="keep",
                    keep_nodes=True,
                    keep_ways=True,
                    keep_relations=True,
                )
                if gdf_chunk is not None and not gdf_chunk.empty:
                    gathered_gdfs.append(gdf_chunk)
            finally:
                if tmp_scene_path.exists():
                    tmp_scene_path.unlink(missing_ok=True)

        if not gathered_gdfs:
            return gpd.GeoDataFrame(geometry=[], crs=target_crs)

        combined = pd.concat(gathered_gdfs, ignore_index=True)
        if "id" in combined.columns:
            combined = combined.drop_duplicates(subset=["id"])

        if combined.crs is None:
            combined = combined.set_crs("EPSG:4326")

        return (
            combined.to_crs(target_crs)
            if str(combined.crs) != target_crs
            else combined
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="resolve historical osm regions and visualize coverage for a target bbox"
    )
    # spatial bounding box parameters
    parser.add_argument("--west", type=float, default=-139.3, help="western longitude boundary")
    parser.add_argument("--south", type=float, default=48.2, help="southern latitude boundary")
    parser.add_argument("--east", type=float, default=-113.0, help="eastern longitude boundary")
    parser.add_argument("--north", type=float, default=60.2, help="northern latitude boundary")

    # optional jurisdiction whitelist filters
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="list of allowed country names or iso codes (e.g. --countries canada us or --countries CA US)",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="list of specific allowed regional ids (e.g. --regions british-columbia alberta)",
    )
    parser.add_argument(
        "--output-map",
        type=str,
        default="data/osm_raw/coverage_map.html",
        help="file path for interactive folium html map output",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="list of region ids or slugs to exclude (e.g. --exclude islas-baleares ceuta andorra)",
    )
    parser.add_argument(
        "--clip-to-countries",
        action="store_true",
        help="clip bounding box to exact country boundary geometries from index",
    )
    parser.add_argument(
        "--polygon-file",
        type=str,
        default=None,
        help="optional geojson/gpkg vector file representing exact adaptive boundaries",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="automatically download all resolved .osh.pbf files into cache_dir",
    )

    args = parser.parse_args()

    args = parser.parse_args()

    # initialize extractor with requested jurisdiction constraints
    extractor = HistoricalOSMExtractor(
        allowed_countries=args.countries,
        allowed_regions=args.regions,
        exclude_regions=args.exclude,
    )

    test_bbox = (args.west, args.south, args.east, args.north)

    print(
        f"resolving regions for bbox: {test_bbox} | "
        f"countries: {args.countries} | "
        f"regions: {args.regions} | "
        f"exclude: {args.exclude}"
    )

    resolved = resolve_finest_covering_regions(
        extractor.index_gdf,
        bbox=test_bbox,
        allowed_countries=args.countries,
        allowed_regions=args.regions,
        exclude_regions=args.exclude,
        clip_to_allowed_countries=args.clip_to_countries,
    )

    print(f"\nresolved {len(resolved)} covering regions:")
    for reg_id, url in resolved:
        print(f"  - {reg_id}")

    # generate interactive visual verification map
    map_file = plot_bbox_coverage(
        index_gdf=extractor.index_gdf,
        bbox=test_bbox,
        resolved_regions=resolved,
        output_html=args.output_map,
    )
    # automatically download missing historical dumps if requested
    if args.download:
        payload = [
            {"region_id": r_id, "history_url": r_url}
            for r_id, r_url in resolved
        ]
        print(f"\nstarting download of {len(payload)} historical dumps into '{extractor.cache_dir}'...")
        downloaded = extractor.download_required_dumps(payload)
        print(f"\nall {len(downloaded)} files are ready for extraction.")