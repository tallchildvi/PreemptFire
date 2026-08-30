from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from filelock import FileLock
import geopandas as gpd
import pandas as pd
from pyrosm import OSM
import requests
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


def load_geofabrik_index(cache_dir: Path) -> gpd.GeoDataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "geofabrik_index.json"

    if not index_path.exists():
        print("  [OSM] Fetching global Geofabrik spatial index...")
        response = requests.get(GEOFABRIK_INDEX_URL, timeout=30)
        response.raise_for_status()
        index_path.write_text(response.text, encoding="utf-8")

    gdf = gpd.read_file(index_path)
    return gdf[gdf.geometry.notna()].reset_index(drop=True)


def parse_history_url(urls_payload: Any) -> Optional[str]:
    if isinstance(urls_payload, dict):
        return urls_payload.get("history")
    if isinstance(urls_payload, str):
        try:
            parsed = json.loads(urls_payload)
            return parsed.get("history") if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def resolve_minimal_covering_regions(
    index_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    min_overlap_km2: float = 1.0,
) -> List[Tuple[str, str]]:
    query_box = box(*bbox)
    intersecting = index_gdf[index_gdf.intersects(query_box)].copy()

    if intersecting.empty:
        raise ValueError(f"No Geofabrik region covers bounding box {bbox}")

    intersecting["history_url"] = intersecting["urls"].apply(parse_history_url)
    candidates = intersecting[
        intersecting["history_url"].notna() & ~intersecting["id"].str.endswith("-admreg")
    ].copy()

    if candidates.empty:
        raise ValueError(f"No valid historical OSM download links found for bounding box {bbox}")

    candidates_proj = candidates.to_crs("EPSG:3857")
    query_box_proj = gpd.GeoSeries([query_box], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    overlaps = []
    for _, row in candidates_proj.iterrows():
        inter = row.geometry.intersection(query_box_proj)
        overlaps.append(inter.area / 1e6 if inter is not None else 0.0)

    candidates["overlap_km2"] = overlaps
    candidates["total_area_km2"] = candidates_proj.geometry.area / 1e6

    valid = candidates[candidates["overlap_km2"] >= min_overlap_km2].copy()
    if valid.empty:
        valid = candidates.sort_values("overlap_km2", ascending=False).head(1)

    valid = valid.sort_values("total_area_km2", ascending=True)

    selected_regions: List[Tuple[str, str]] = []
    uncovered_geom = query_box

    for _, row in valid.iterrows():
        inter = row.geometry.intersection(uncovered_geom)
        if inter is None or inter.is_empty:
            continue

        selected_regions.append((row["id"], row["history_url"]))
        uncovered_geom = uncovered_geom.difference(row.geometry)

        if uncovered_geom.is_empty or (uncovered_geom.area / query_box.area < 0.001):
            break

    return selected_regions


def ensure_history_dump(region_id: str, history_url: str, cache_dir: Path) -> Path:
    clean_id = region_id.replace("/", "_")
    target_path = cache_dir / f"{clean_id}-history.osh.pbf"
    lock_path = target_path.with_suffix(".lock")

    if target_path.exists():
        return target_path

    cookie = os.getenv("GEOFABRIK_COOKIE")
    if not cookie:
        raise ValueError("GEOFABRIK_COOKIE не знайдено у .env файлі.")

    cache_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=7200):
        if target_path.exists():
            return target_path

        print(f"\n  [OSM] Downloading historical dump '{region_id}'\n        {history_url}")
        tmp_path = target_path.with_suffix(".tmp")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Cookie": cookie.strip('"\''),
        }

        with requests.get(history_url, headers=headers, stream=True, timeout=60) as resp:
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

            tmp_path.rename(target_path)
            print(f"\n  [OSM] Saved to {target_path.name}")

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
    """Cuts exact scene BBox out of provincial PBF using C++ Osmium in ~0.3s."""
    west, south, east, north = bbox
    bbox_tag = f"{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}".replace("-", "m").replace(".", "p")
    scene_pbf_path = cache_dir / f"scene_{source_pbf.stem}_{bbox_tag}.osm.pbf"

    if scene_pbf_path.exists():
        return scene_pbf_path

    cmd = [
        "osmium", "extract",
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
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_gdf = load_geofabrik_index(self.cache_dir)

    def resolve_history_files(
        self, west: float, south: float, east: float, north: float
    ) -> List[Dict[str, Any]]:
        regions = resolve_minimal_covering_regions(self.index_gdf, (west, south, east, north))
        output = []

        for region_id, history_url in regions:
            clean_id = region_id.replace("/", "_")
            local_path = self.cache_dir / f"{clean_id}-history.osh.pbf"
            output.append({
                "region_id": region_id,
                "history_url": history_url,
                "local_path": local_path,
                "downloaded": local_path.exists(),
            })

        return output

    def download_required_dumps(self, resolved_files: List[Dict[str, Any]]) -> List[Path]:
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
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else date.split(" ")[0]
        bbox = (west, south, east, north)
        resolved_files = self.resolve_history_files(west, south, east, north)
        gathered_gdfs: List[gpd.GeoDataFrame] = []

        for entry in resolved_files:
            osh_path = ensure_history_dump(entry["region_id"], entry["history_url"], self.cache_dir)
            
            # 1. Повний зріз провінції на дату (вже згенерований і лежить на SSD)
            snapshot_pbf = get_or_create_snapshot_pbf(osh_path, date_str)

            # 2. C++ миттєво вирізає тільки BBox сцени (~2 MB)
            t_extract = time.perf_counter()
            scene_pbf = extract_scene_bbox_pbf(snapshot_pbf, bbox, self.cache_dir)
            print(f"  [OSM Spatial Crop] Cropped BBox scene in {time.perf_counter() - t_extract:.2f}s")

            # 3. pyrosm читає легкий 2 MB файл миттєво
            t0 = time.perf_counter()
            osm = OSM(str(scene_pbf))
            gdf_chunk = osm.get_data_by_custom_criteria(
                custom_filter=OSM_EXTRACTION_FILTER,
                filter_type="keep",
                keep_nodes=True,
                keep_ways=True,
                keep_relations=False,
            )
            duration = time.perf_counter() - t0
            count = len(gdf_chunk) if gdf_chunk is not None else 0
            print(f"  [OSM] Parsed {count:,} features from scene in {duration:.2f}s")

            if gdf_chunk is not None and not gdf_chunk.empty:
                gathered_gdfs.append(gdf_chunk)

        if not gathered_gdfs:
            return gpd.GeoDataFrame(geometry=[], crs=target_crs)

        combined = pd.concat(gathered_gdfs, ignore_index=True)
        if "id" in combined.columns:
            combined = combined.drop_duplicates(subset=["id"])

        if combined.crs is None:
            combined = combined.set_crs("EPSG:4326")

        return combined.to_crs(target_crs) if str(combined.crs) != target_crs else combined


if __name__ == "__main__":
    extractor = HistoricalOSMExtractor()

    bc_west, bc_south, bc_east, bc_north = -120.6, 50.4, -119.9, 50.9
    sample_date = "2021-08-15"

    print(f"\n1. Resolving historical dumps for bbox: [{bc_west}, {bc_south}, {bc_east}, {bc_north}]")
    files_info = extractor.resolve_history_files(bc_west, bc_south, bc_east, bc_north)

    print("\n2. Ensuring historical .osh.pbf dump is present...")
    extractor.download_required_dumps(files_info)

    print(f"\n3. Extracting point-in-time features for {sample_date}...")
    t0 = time.perf_counter()
    gdf = extractor.extract_features_for_date(
        date=sample_date,
        west=bc_west,
        south=bc_south,
        east=bc_east,
        north=bc_north,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nDone: Extracted {len(gdf):,} historical geometries in {elapsed:.2f}s")
    if not gdf.empty:
        print("Feature layers detected:", [c for c in gdf.columns if c != "geometry"][:8])
        print(gdf[["geometry"] + [c for c in ["highway", "waterway", "natural"] if c in gdf.columns]].head(3))