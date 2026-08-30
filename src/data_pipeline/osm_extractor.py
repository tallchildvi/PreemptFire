from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

from filelock import FileLock
import geopandas as gpd
import pandas as pd
from pyrosm import OSM
import requests
from shapely.geometry import box

from src.config import BASE_DIR

DEFAULT_CACHE_DIR = BASE_DIR / "data" / "osm_raw"
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


def parse_pbf_url(urls_payload: Any) -> Optional[str]:
    if isinstance(urls_payload, dict):
        return urls_payload.get("pbf")
    if isinstance(urls_payload, str):
        try:
            parsed = json.loads(urls_payload)
            return parsed.get("pbf") if isinstance(parsed, dict) else None
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

    intersecting["pbf_url"] = intersecting["urls"].apply(parse_pbf_url)
    candidates = intersecting[
        intersecting["pbf_url"].notna() & ~intersecting["id"].str.endswith("-admreg")
    ].copy()

    if candidates.empty:
        raise ValueError(f"No valid PBF download links found for bounding box {bbox}")

    candidates_proj = candidates.to_crs("EPSG:3857")
    query_box_proj = gpd.GeoSeries([query_box], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    overlaps_km2 = [
        (row.geometry.intersection(query_box_proj).area / 1e6)
        if row.geometry.intersection(query_box_proj) is not None
        else 0.0
        for _, row in candidates_proj.iterrows()
    ]
    candidates["overlap_km2"] = overlaps_km2
    candidates["total_area_km2"] = candidates_proj.geometry.area / 1e6

    valid = candidates[candidates["overlap_km2"] >= min_overlap_km2].copy()
    if valid.empty:
        valid = candidates.sort_values("overlap_km2", ascending=False).head(1)

    valid = valid.sort_values("total_area_km2", ascending=True)

    selected_regions: List[Tuple[str, str]] = []
    uncovered_geom = query_box

    for _, row in valid.iterrows():
        intersection = row.geometry.intersection(uncovered_geom)
        if intersection is None or intersection.is_empty:
            continue

        selected_regions.append((row["id"], row["pbf_url"]))
        uncovered_geom = uncovered_geom.difference(row.geometry)

        if uncovered_geom.is_empty or (uncovered_geom.area / query_box.area < 0.001):
            break

    return selected_regions


def stream_download_progress(blocks: int, block_size: int, total_size: int) -> None:
    downloaded = blocks * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        sys.stdout.write(f"\r  [Download] {mb_down:.1f}/{mb_total:.1f} MB ({percent:.1f}%)")
        sys.stdout.flush()


def ensure_pbf_dump(region_id: str, pbf_url: str, cache_dir: Path) -> Path:
    clean_id = region_id.replace("/", "_")
    target_path = cache_dir / f"{clean_id}-latest.osm.pbf"
    lock_path = target_path.with_suffix(".lock")

    if target_path.exists():
        return target_path

    cache_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=3600):
        if target_path.exists():
            return target_path

        print(f"\n  [OSM] Downloading '{region_id}'\n        {pbf_url}")
        tmp_path = target_path.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(pbf_url, tmp_path, reporthook=stream_download_progress)
            tmp_path.rename(target_path)
            print(f"\n  [OSM] Saved to {target_path.name}")
        except Exception as err:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed downloading {pbf_url}: {err}") from err

    return target_path


class UniversalOSMExtractor:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_gdf = load_geofabrik_index(self.cache_dir)

    def resolve_pbf_files(
        self, west: float, south: float, east: float, north: float
    ) -> List[Dict[str, Any]]:
        regions = resolve_minimal_covering_regions(self.index_gdf, (west, south, east, north))
        output = []

        for region_id, pbf_url in regions:
            clean_id = region_id.replace("/", "_")
            local_path = self.cache_dir / f"{clean_id}-latest.osm.pbf"
            output.append({
                "region_id": region_id,
                "pbf_url": pbf_url,
                "local_path": local_path,
                "downloaded": local_path.exists(),
            })

        return output

    def download_required_dumps(self, resolved_files: List[Dict[str, Any]]) -> List[Path]:
        paths = []
        for entry in resolved_files:
            pbf_path = ensure_pbf_dump(entry["region_id"], entry["pbf_url"], self.cache_dir)
            paths.append(pbf_path)
        return paths

    def _parse_single_pbf(
        self, pbf_path: Path, bbox: Tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        osm = OSM(str(pbf_path), bounding_box=list(bbox))
        try:
            gdf = osm.get_data_by_custom_criteria(
                custom_filter=OSM_EXTRACTION_FILTER,
                filter_type="keep",
                keep_nodes=True,
                keep_ways=True,
                keep_relations=False,
            )
        except Exception as err:
            print(f"  [OSM] Warning: parse error on {pbf_path.name}: {err}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if gdf is None or gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        return gdf

    def extract_features_for_bbox(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        target_crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        bbox = (west, south, east, north)
        resolved_files = self.resolve_pbf_files(west, south, east, north)
        gathered_gdfs: List[gpd.GeoDataFrame] = []

        for entry in resolved_files:
            pbf_path = ensure_pbf_dump(entry["region_id"], entry["pbf_url"], self.cache_dir)
            t_start = time.perf_counter()
            gdf_chunk = self._parse_single_pbf(pbf_path, bbox)
            duration = time.perf_counter() - t_start

            print(f"  [OSM] {pbf_path.name} -> {len(gdf_chunk):,} elements ({duration:.2f}s)")
            if not gdf_chunk.empty:
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

    extractor = UniversalOSMExtractor()

    bc_west, bc_south, bc_east, bc_north = -139.0, 49.0, -114.0, 60.0

    print(f"\nChecking required OSM dumps for bbox: [{bc_west}, {bc_south}, {bc_east}, {bc_north}]")
    files_info = extractor.resolve_pbf_files(bc_west, bc_south, bc_east, bc_north)

    for item in files_info:
        status = "EXISTS" if item["downloaded"] else "MISSING"
        print(f"   - Region: {item['region_id']}")
        print(f"     Status: {status}")
        print(f"     File:   {item['local_path'].name}")
        print(f"     URL:    {item['pbf_url']}\n")