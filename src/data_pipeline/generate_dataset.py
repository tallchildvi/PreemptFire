import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Dict, List, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_pipeline.hdf5_writer import HDF5Writer
from src.data_pipeline.patch_extractor import PatchExtractor
from src.data_pipeline.single_scene_collector import SingleSceneCollector


class DatasetGenerator:
    """Parallel batch processor creating isolated HDF5 shards with robust SQLite recovery."""

    def __init__(
        self,
        points_csv_path: str,
        output_shards_dir: str = "data/processed/shards",
        patch_size: int = 256,
        stride: int = 256,
        max_invalid_ratio: float = 0.20,
        osm_mode: str = "local_history",
        max_workers: int = 4,
    ):
        self.points_csv_path = points_csv_path
        self.output_shards_dir = Path(output_shards_dir)
        self.output_shards_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        self.db_path = self.output_shards_dir.parent / "dataset_checkpoint.db"
        self._db_lock = threading.Lock()
        self._init_db()
        self._recover_stuck_scenes()

        # Collector is thread-safe (mostly network I/O)
        self.collector = SingleSceneCollector(osm_mode=osm_mode)
        self.patch_size = patch_size
        self.stride = stride
        self.max_invalid_ratio = max_invalid_ratio

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=60.0)

    def _init_db(self):
        """Creates checkpoint table if not exists."""
        with self._db_lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scenes_tracker (
                    scene_id TEXT PRIMARY KEY,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    target_date TEXT NOT NULL,
                    is_fire INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    patches_count INTEGER DEFAULT 0,
                    error_msg TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()

    def _recover_stuck_scenes(self):
        """Resets any scenes that were left in 'PROCESSING' state due to a power outage."""
        with self._db_lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE scenes_tracker SET status = 'PENDING' WHERE status = 'PROCESSING'")
            recovered = cursor.rowcount
            conn.commit()
            if recovered > 0:
                print(f"[RECOVERY] Reverted {recovered} interrupted scenes back to PENDING.")

    def sync_from_csv(self, df: pd.DataFrame):
        """Registers new points from CSV into tracker."""
        with self._db_lock, self._get_connection() as conn:
            cursor = conn.cursor()
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            for _, row in df.iterrows():
                lat = float(row.get("latitude", row.get("lat")))
                lon = float(row.get("longitude", row.get("lon")))
                raw_date = row.get("acq_date", row.get("target_date"))
                date_str = str(raw_date).split(" ")[0].strip()

                is_fire = int(row.get("is_fire", 1))
                scene_id = f"SCENE_{lat:.4f}_{lon:.4f}_{date_str.replace('-', '')}"

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO scenes_tracker 
                    (scene_id, lat, lon, target_date, is_fire, status, patches_count, error_msg, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'PENDING', 0, NULL, ?)
                    """,
                    (scene_id, lat, lon, date_str, is_fire, now),
                )
            conn.commit()

    def _update_status(self, scene_id: str, status: str, patches_count: int = 0, error_msg: Optional[str] = None):
        """Thread-safe status update."""
        with self._db_lock, self._get_connection() as conn:
            cursor = conn.cursor()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                UPDATE scenes_tracker 
                SET status = ?, patches_count = ?, error_msg = ?, updated_at = ?
                WHERE scene_id = ?
                """,
                (status, patches_count, error_msg, now, scene_id),
            )
            conn.commit()

    def _process_single_scene(self, scene: dict) -> str:
        """Worker function that fully processes one scene and writes it to its own HDF5 shard."""
        import shutil
        free_gb = shutil.disk_usage(self.output_shards_dir).free / (1024**3)
        if free_gb < 5.0:
            return f"[{scene['scene_id']}] STOPPED: Low disk space ({free_gb:.1f} GB free)"

        scene_id = scene["scene_id"]
        lat, lon = scene["lat"], scene["lon"]
        target_date = scene["target_date"]
        is_fire = scene["is_fire"]

        self._update_status(scene_id, status="PROCESSING")
        t_start = time.perf_counter()

        try:
            sample = self.collector.collect_sample(
                lat=lat, lon=lon, target_date=target_date, is_fire=is_fire
            )

            if sample is None:
                self._update_status(scene_id, status="SKIPPED_NO_DATA", error_msg="Missing S2 optical pair or heavy clouds")
                return f"[{scene_id}] SKIPPED (No data)"

            extractor = PatchExtractor(
                patch_size=self.patch_size,
                stride=self.stride,
                max_invalid_ratio=self.max_invalid_ratio,
            )
            patches = list(extractor.extract_patches(sample, scene_id=scene_id))

            if not patches:
                self._update_status(scene_id, status="SKIPPED_EMPTY_PATCHES", error_msg="0 valid patches passed quality gate")
                return f"[{scene_id}] SKIPPED (0 valid patches)"

            shard_path = self.output_shards_dir / f"{scene_id}.h5"
            shard_path.unlink(missing_ok=True)  # захист від дублювання при ретраях
            writer = HDF5Writer(output_path=shard_path, patch_size=self.patch_size)
            written_count = writer.write_patches_batch(patches)

            elapsed = time.perf_counter() - t_start
            self._update_status(scene_id, status="COMPLETED", patches_count=written_count)
            return f"[{scene_id}] SUCCESS: {written_count} patches in {elapsed:.1f}s"

        except Exception as e:
            self._update_status(scene_id, status="FAILED", error_msg=str(e))
            return f"[{scene_id}] FAILED: {str(e)}"

        
    def run(self, max_scenes: Optional[int] = None):
        """Executes parallel batch collection."""
        df_points = pd.read_csv(self.points_csv_path)
        self.sync_from_csv(df_points)

        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # FAILED are retried automatically, PENDING are new/recovered
            cursor.execute(
                """
                SELECT scene_id, lat, lon, target_date, is_fire 
                FROM scenes_tracker 
                WHERE status IN ('PENDING', 'FAILED')
                ORDER BY target_date ASC, scene_id ASC
                """
            )
            pending_scenes = [dict(r) for r in cursor.fetchall()]

        if max_scenes:
            pending_scenes = pending_scenes[:max_scenes]

        print(f"\n--- Found {len(pending_scenes)} scenes pending processing ---")
        print(f"--- Starting parallel execution with {self.max_workers} workers ---")

        processed_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map futures to scenes for parallel execution
            futures = {executor.submit(self._process_single_scene, scene): scene for scene in pending_scenes}

            for future in as_completed(futures):
                processed_count += 1
                result_msg = future.result()
                print(f"({processed_count}/{len(pending_scenes)}) {result_msg}")


if __name__ == "__main__":
    csv_input_path = os.path.join("data", "interim", "points_dataset_thinned.csv")
    output_shards_dir = os.path.join("data", "processed", "shards")

    if not os.path.exists(csv_input_path):
        raise FileNotFoundError(f"input csv file not found: {csv_input_path}")

    print("--- initializing resilient parallel generator ---")
    generator = DatasetGenerator(
        points_csv_path=csv_input_path,
        output_shards_dir=output_shards_dir,
        patch_size=256,
        stride=256,
        max_invalid_ratio=0.20,
        osm_mode="local_history",
        max_workers=2,
    )
    generator.run(max_scenes=None)