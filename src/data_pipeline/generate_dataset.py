import os
import sqlite3
import time
from typing import Dict, List, Optional
import pandas as pd

from src.data_pipeline.hdf5_writer import HDF5Writer
from src.data_pipeline.patch_extractor import PatchExtractor
from src.data_pipeline.single_scene_collector import SingleSceneCollector


class DatasetGenerator:
    """batch processor creating a unified hdf5 dataset with sqlite-based state recovery."""

    def __init__(
        self,
        points_csv_path: str,
        output_h5_path: str = "data/processed/wildfire_dataset.h5",
        patch_size: int = 256,
        stride: int = 256,
        max_invalid_ratio: float = 0.20,
    ):
        self.points_csv_path = points_csv_path
        self.output_h5_path = output_h5_path
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

        self.db_path = os.path.join(os.path.dirname(output_h5_path), "dataset_checkpoint.db")
        self._init_db()

        self.collector = SingleSceneCollector()
        self.extractor = PatchExtractor(
            patch_size=patch_size,
            stride=stride,
            max_invalid_ratio=max_invalid_ratio,
        )
        self.writer = HDF5Writer(self.output_h5_path, patch_size=patch_size)

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self):
        """creates checkpoint table if not exists."""
        with self._get_connection() as conn:
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

    def sync_from_csv(self, df: pd.DataFrame):
        """registers new points from csv into tracker supporting custom schema mappings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            for _, row in df.iterrows():
                # support both latitude/longitude and lat/lon schemas
                lat = float(row.get("latitude", row.get("lat")))
                lon = float(row.get("longitude", row.get("lon")))

                # support both acq_date and target_date
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

    def run(self, max_scenes: Optional[int] = None):
        """executes batch collection and atomic streaming into a single hdf5 file."""
        df_points = pd.read_csv(self.points_csv_path)
        self.sync_from_csv(df_points)

        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT scene_id, lat, lon, target_date, is_fire 
                FROM scenes_tracker 
                WHERE status IN ('PENDING', 'FAILED')
                ORDER BY is_fire DESC, target_date ASC
                """
            )
            pending_scenes = [dict(r) for r in cursor.fetchall()]

        if max_scenes:
            pending_scenes = pending_scenes[:max_scenes]

        print(f"found {len(pending_scenes)} scenes pending processing.")

        for idx, scene in enumerate(pending_scenes, 1):
            scene_id = scene["scene_id"]
            lat, lon = scene["lat"], scene["lon"]
            target_date = scene["target_date"]
            is_fire = scene["is_fire"]

            print(f"\n[{idx}/{len(pending_scenes)}] processing {scene_id} | fire={is_fire}")
            t_start = time.perf_counter()

            try:
                sample = self.collector.collect_sample(
                    lat=lat, lon=lon, target_date=target_date, is_fire=is_fire
                )

                if sample is None:
                    self._update_status(scene_id, status="FAILED", error_msg="collector returned none")
                    continue

                patches = list(self.extractor.extract_patches(sample, scene_id=scene_id))

                if not patches:
                    self._update_status(scene_id, status="COMPLETED", patches_count=0, error_msg="0 valid patches")
                    continue

                written_count = self.writer.write_patches_batch(patches)
                elapsed = time.perf_counter() - t_start
                print(f"saved {written_count} patches in {elapsed:.2f}s")

                self._update_status(scene_id, status="COMPLETED", patches_count=written_count)

            except Exception as e:
                print(f"error on {scene_id}: {str(e)}")
                self._update_status(scene_id, status="FAILED", error_msg=str(e))

    def _update_status(self, scene_id: str, status: str, patches_count: int = 0, error_msg: Optional[str] = None):
        with self._get_connection() as conn:
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


if __name__ == "__main__":
    from src.data_pipeline.hdf5_writer import inspect_h5_structure

    csv_input_path = os.path.join("data", "interim", "points_dataset_thinned.csv")
    output_h5_path = os.path.join("data", "processed", "wildfire_dataset.h5")

    if not os.path.exists(csv_input_path):
        raise FileNotFoundError(f"input csv file not found: {csv_input_path}")

    print("--- initializing dataset generator ---")
    generator = DatasetGenerator(
        points_csv_path=csv_input_path,
        output_h5_path=output_h5_path,
        patch_size=256,
        stride=256,
        max_invalid_ratio=0.20,
    )

    print("--- processing first 5 scenes ---")
    generator.run(max_scenes=5)

    if os.path.exists(output_h5_path):
        print("\n--- inspecting hdf5 structure ---")
        inspect_h5_structure(output_h5_path)