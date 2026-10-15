import os
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time
from typing import Optional
import h5py
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
        max_attempts: int = 3,
    ):
        self.points_csv_path = points_csv_path
        self.output_shards_dir = Path(output_shards_dir)
        self.output_shards_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.max_attempts = max_attempts

        self.db_path = self.output_shards_dir.parent / "dataset_checkpoint.db"
        self._db_lock = threading.Lock()
        self._init_db()
        self._cleanup_tmp_files()
        self._recover_stuck_scenes()
        self._verify_completed_scenes()

        # Collector is thread-safe (mostly network I/O)
        self.collector = SingleSceneCollector(osm_mode=osm_mode)
        self.patch_size = patch_size
        self.stride = stride
        self.max_invalid_ratio = max_invalid_ratio

    # SQLite helpers

    @contextmanager
    def _db(self):
        """Opens a durable connection, commits/rolls back the transaction and ALWAYS closes it."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            with conn:  # commit on success, rollback on exception
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Creates checkpoint table if not exists (and migrates older DBs)."""
        with self._db_lock, self._db() as conn:
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
                    updated_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Migration for databases created before the 'attempts' column existed
            columns = {row[1] for row in cursor.execute("PRAGMA table_info(scenes_tracker)")}
            if "attempts" not in columns:
                cursor.execute("ALTER TABLE scenes_tracker ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
    # Recovery
    
    def _cleanup_tmp_files(self):
        """Removes half-written temporary shards left after a crash / power outage."""
        removed = 0
        for tmp_file in self.output_shards_dir.glob("*.h5.tmp"):
            try:
                tmp_file.unlink()
                removed += 1
            except OSError as e:
                print(f"[RECOVERY] Could not remove {tmp_file.name}: {e}")
        if removed > 0:
            print(f"[RECOVERY] Removed {removed} incomplete temporary shard file(s).")

    def _recover_stuck_scenes(self):
        """Resets any scenes that were left in 'PROCESSING' state due to a power outage."""
        with self._db_lock, self._db() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE scenes_tracker SET status = 'PENDING' WHERE status = 'PROCESSING'")
            recovered = cursor.rowcount
        if recovered > 0:
            print(f"[RECOVERY] Reverted {recovered} interrupted scenes back to PENDING.")

    @staticmethod
    def _is_valid_h5(path: Path) -> bool:
        """Checks that the file exists and can be opened as a valid HDF5 file."""
        if not path.exists():
            return False
        try:
            with h5py.File(path, "r"):
                return True
        except Exception:
            return False

    def _verify_completed_scenes(self):
        """Re-queues COMPLETED scenes whose shard is missing or corrupted on disk."""
        with self._db_lock, self._db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT scene_id FROM scenes_tracker WHERE status = 'COMPLETED'")
            completed_ids = [row[0] for row in cursor.fetchall()]

            broken = []
            for scene_id in completed_ids:
                shard_path = self.output_shards_dir / f"{scene_id}.h5"
                if not self._is_valid_h5(shard_path):
                    broken.append(scene_id)
                    if shard_path.exists():
                        try:
                            shard_path.unlink()  # remove corrupted file so it gets regenerated
                        except OSError:
                            pass

            if broken:
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                cursor.executemany(
                    """
                    UPDATE scenes_tracker
                    SET status = 'PENDING', patches_count = 0,
                        error_msg = 'Shard missing or corrupted, re-queued', updated_at = ?
                    WHERE scene_id = ?
                    """,
                    [(now, scene_id) for scene_id in broken],
                )
                print(f"[RECOVERY] Re-queued {len(broken)} COMPLETED scenes with missing/corrupted shards.")

    
    # Tracker operations
    
    def sync_from_csv(self, df: pd.DataFrame):
        """Registers new points from CSV into tracker."""
        with self._db_lock, self._db() as conn:
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
                    (scene_id, lat, lon, target_date, is_fire, status, patches_count, error_msg, updated_at, attempts)
                    VALUES (?, ?, ?, ?, ?, 'PENDING', 0, NULL, ?, 0)
                    """,
                    (scene_id, lat, lon, date_str, is_fire, now),
                )

    def _update_status(
        self,
        scene_id: str,
        status: str,
        patches_count: int = 0,
        error_msg: Optional[str] = None,
        increment_attempts: bool = False,
    ):
        """Thread-safe status update."""
        with self._db_lock, self._db() as conn:
            cursor = conn.cursor()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                UPDATE scenes_tracker 
                SET status = ?, patches_count = ?, error_msg = ?, updated_at = ?,
                    attempts = attempts + ?
                WHERE scene_id = ?
                """,
                (status, patches_count, error_msg, now, int(increment_attempts), scene_id),
            )

    
    # Durable file helpers
    
    @staticmethod
    def _fsync_file(path: Path):
        """Forces file contents to physical disk before it is renamed into place."""
        with open(path, "r+b") as f:
            os.fsync(f.fileno())

    @staticmethod
    def _fsync_dir(path: Path):
        """Persists the directory entry after rename (best effort, not supported on Windows)."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    
    # Worker
    
    def _process_single_scene(self, scene: dict) -> str:
        """Worker function that fully processes one scene and writes it to its own HDF5 shard."""
        scene_id = scene["scene_id"]
        lat, lon = scene["lat"], scene["lon"]
        target_date = scene["target_date"]
        is_fire = scene["is_fire"]

        shard_path = self.output_shards_dir / f"{scene_id}.h5"
        tmp_path = self.output_shards_dir / f"{scene_id}.h5.tmp"

        self._update_status(scene_id, status="PROCESSING")
        t_start = time.perf_counter()

        try:
            # 1. Collect Multi-modal Data
            sample = self.collector.collect_sample(
                lat=lat, lon=lon, target_date=target_date, is_fire=is_fire
            )

            if sample is None:
                self._update_status(scene_id, status="SKIPPED_NO_DATA", error_msg="Missing S2 optical pair or heavy clouds")
                return f"[{scene_id}] SKIPPED (No data)"

            # 2. Extract Patches (Extractor is lightweight, safe to instantiate per thread)
            extractor = PatchExtractor(
                patch_size=self.patch_size,
                stride=self.stride,
                max_invalid_ratio=self.max_invalid_ratio,
            )
            patches = list(extractor.extract_patches(sample, scene_id=scene_id))

            if not patches:
                self._update_status(scene_id, status="SKIPPED_EMPTY_PATCHES", error_msg="0 valid patches passed quality gate")
                return f"[{scene_id}] SKIPPED (0 valid patches)"

            # 3. Write to a temporary file first, then atomically move it into place.
            #    A power loss mid-write can never leave a broken file under the final name.
            if tmp_path.exists():
                tmp_path.unlink()

            writer = HDF5Writer(output_path=tmp_path, patch_size=self.patch_size)
            written_count = writer.write_patches_batch(patches)
            # NOTE: HDF5Writer must have closed the file by this point
            # (close() inside write_patches_batch or a context manager).

            self._fsync_file(tmp_path)
            os.replace(tmp_path, shard_path)  # atomic on the same filesystem
            self._fsync_dir(self.output_shards_dir)

            elapsed = time.perf_counter() - t_start
            self._update_status(scene_id, status="COMPLETED", patches_count=written_count)
            return f"[{scene_id}] SUCCESS: {written_count} patches in {elapsed:.1f}s"

        except Exception as e:
            # Do not leave a partial temp file behind
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            self._update_status(scene_id, status="FAILED", error_msg=str(e), increment_attempts=True)
            return f"[{scene_id}] FAILED: {str(e)}"

    
    # Runner
    
    def run(self, max_scenes: Optional[int] = None):
        """Executes parallel batch collection."""
        df_points = pd.read_csv(self.points_csv_path)
        self.sync_from_csv(df_points)

        with self._db() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # PENDING are new/recovered; FAILED are retried until max_attempts is reached
            cursor.execute(
                """
                SELECT scene_id, lat, lon, target_date, is_fire 
                FROM scenes_tracker 
                WHERE status = 'PENDING'
                   OR (status = 'FAILED' AND attempts < ?)
                ORDER BY is_fire DESC, target_date ASC
                """,
                (self.max_attempts,),
            )
            pending_scenes = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                "SELECT COUNT(*) FROM scenes_tracker WHERE status = 'FAILED' AND attempts >= ?",
                (self.max_attempts,),
            )
            permanently_failed = cursor.fetchone()[0]

        if max_scenes:
            pending_scenes = pending_scenes[:max_scenes]

        print(f"\n--- Found {len(pending_scenes)} scenes pending processing ---")
        if permanently_failed:
            print(f"--- {permanently_failed} scenes exceeded {self.max_attempts} attempts and are skipped ---")
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
        max_workers=4, # 4 is optimal to avoid triggering NASA/MS planetary computer rate limits
        max_attempts=3,
    )

    generator.run(max_scenes=10)