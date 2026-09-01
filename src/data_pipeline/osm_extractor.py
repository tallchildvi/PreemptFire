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

# Aggregate/macro regions that must never be downloaded
EXCLUDED_MACRO_REGIONS = {
    "us", "us-west", "us-pacific", "us-midwest", "us-south", "us-northeast",
    "canada", "north-america", "russia", "europe", "central-america", "mexico"
}


def load_geofabrik_index(cache_dir: Path) -> gpd.GeoDataFrame:
    """Fetches and caches the global Geofabrik spatial index."""
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
    """Extracts the internal history PBF download link."""
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
    min_overlap_km2: float = 25.0,
) -> List[Tuple[str, str]]:
    """Identifies the minimal set of leaf sub-regions covering a target BBox, strictly ignoring macro-regions."""
    query_box = box(*bbox)
    intersecting = index_gdf[index_gdf.intersects(query_box)].copy()

    if intersecting.empty:
        raise ValueError(f"No Geofabrik region covers bounding box {bbox}")

    # 1. Identify all parent regions that contain sub-regions
    parent_ids = set(index_gdf["parent"].dropna().unique())

    intersecting["history_url"] = intersecting["urls"].apply(parse_history_url)

    # 2. Keep ONLY genuine leaf regions (provinces / individual states)
    candidates = intersecting[
        intersecting["history_url"].notna()
        & ~intersecting["id"].str.endswith("-admreg")
        & ~intersecting["id"].isin(parent_ids)
        & ~intersecting["id"].isin(EXCLUDED_MACRO_REGIONS)
    ].copy()

    if candidates.empty:
        raise ValueError(f"No valid historical OSM leaf dumps found for bounding box {bbox}")

    candidates_proj = candidates.to_crs("EPSG:3857")
    query_box_proj = gpd.GeoSeries([query_box], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    overlaps = []
    for _, row in candidates_proj.iterrows():
        inter = row.geometry.intersection(query_box_proj)
        overlaps.append(inter.area / 1e6 if inter is not None else 0.0)

    candidates["overlap_km2"] = overlaps
    valid = candidates[candidates["overlap_km2"] >= min_overlap_km2].copy()

    if valid.empty:
        valid = candidates.sort_values("overlap_km2", ascending=False).head(1)

    # Return all unique intersecting leaf regions directly
    selected_regions = [
        (row["id"], row["history_url"])
        for _, row in valid.sort_values("overlap_km2", ascending=False).iterrows()
    ]
    return selected_regions


def ensure_history_dump(region_id: str, history_url: str, cache_dir: Path) -> Path:
    """Downloads authenticated historical .osh.pbf file with strict auth and size validation."""
    clean_id = region_id.replace("/", "_")
    target_path = cache_dir / f"{clean_id}-history.osh.pbf"
    lock_path = target_path.with_suffix(".lock")

    if target_path.exists() and target_path.stat().st_size > 10_485_760:
        return target_path

    cookie = os.getenv("GEOFABRIK_COOKIE")
    if not cookie:
        raise ValueError("GEOFABRIK_COOKIE is missing in .env file.")

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
                raise RuntimeError(
                    f"Downloaded file {target_path.name} is too small (<10MB). "
                    "Download failed or cookie expired."
                )

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
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_gdf = load_geofabrik_index(self.cache_dir)

    def resolve_history_files(
        self, west: float, south: float, east: float, north: float
    ) -> List[Dict[str, Any]]:
        """Resolves covering regions for a BBox and checks local existence."""
        regions = resolve_minimal_covering_regions(self.index_gdf, (west, south, east, north))
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
        """Downloads all missing regional .osh.pbf files."""
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
        """Extracts and parses historical features for a given date and BBox."""
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else date.split(" ")[0]
        bbox = (west, south, east, north)
        resolved_files = self.resolve_history_files(west, south, east, north)
        gathered_gdfs: List[gpd.GeoDataFrame] = []

        for entry in resolved_files:
            osh_path = ensure_history_dump(entry["region_id"], entry["history_url"], self.cache_dir)
            snapshot_pbf = get_or_create_snapshot_pbf(osh_path, date_str)

            t_extract = time.perf_counter()
            scene_pbf = extract_scene_bbox_pbf(snapshot_pbf, bbox, self.cache_dir)
            print(f"  [OSM Spatial Crop] Cropped scene in {time.perf_counter() - t_extract:.2f}s")

            t0 = time.perf_counter()
            osm = OSM(str(scene_pbf))
            gdf_chunk = osm.get_data_by_custom_criteria(
                custom_filter=OSM_EXTRACTION_FILTER,
                filter_type="keep",
                keep_nodes=True,
                keep_ways=True,
                keep_relations=True,
            )
            duration = time.perf_counter() - t0
            count = len(gdf_chunk) if gdf_chunk is not None else 0
            print(f"  [OSM] Parsed {count:,} features in {duration:.2f}s")

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

    west, south, east, north = -139.3, 48.2, -113.0, 60.2

    print(f"1. Resolving covering historical dumps for BBox: [{west}, {south}, {east}, {north}]")
    files_info = extractor.resolve_history_files(west, south, east, north)

    print(f"\nFound {len(files_info)} regional dump(s):")
    for item in files_info:
        status = "CACHED" if item["downloaded"] else "PENDING_DOWNLOAD"
        print(f"  - [{status}] {item['region_id']} -> {item['local_path'].name}")

    print("\n2. Downloading missing .osh.pbf dump files...")
    downloaded = extractor.download_required_dumps(files_info)

    print(f"\nAll regional files are ready in '{DEFAULT_CACHE_DIR}':")
    for path in downloaded:
        print(f"  - {path.name} ({path.stat().st_size / (1024 * 1024):.1f} MB)")