from datetime import datetime
import os
import sqlite3
import time
from typing import Dict, List, Optional
import pandas as pd

from src.data_pipeline.hdf5_writer import HDF5Writer
from src.data_pipeline.patch_extractor import PatchExtractor
from src.data_pipeline.single_scene_collector import SingleSceneCollector


class DatasetProgressTracker:
    """manages sqlite-based state and checkpointing for scene processing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

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
                    split TEXT NOT NULL,
                    status TEXT NOT NULL,
                    patches_count INTEGER DEFAULT 0,
                    error_msg TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()

    def sync_from_csv(self, df: pd.DataFrame, default_split: str = "train"):
        """registers new points from csv into tracker without overwriting completed ones."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            for _, row in df.iterrows():
                lat = float(row["lat"])
                lon = float(row["lon"])
                date_str = str(row["target_date"]).split(" ")[0]
                is_fire = int(row.get("is_fire", 1))
                split = str(row.get("split", default_split)).lower()
                scene_id = f"SCENE_{lat:.4f}_{lon:.4f}_{date_str.replace('-', '')}"

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO scenes_tracker 
                    (scene_id, lat, lon, target_date, is_fire, split, status, patches_count, error_msg, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'PENDING', 0, NULL, ?)
                    """,
                    (scene_id, lat, lon, date_str, is_fire, split, now),
                )
            conn.commit()

    def get_pending_scenes(self) -> List[Dict]:
        """retrieves all scenes requiring processing or retries."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT scene_id, lat, lon, target_date, is_fire, split 
                FROM scenes_tracker 
                WHERE status IN ('PENDING', 'FAILED')
                ORDER BY is_fire DESC, target_date ASC
                """
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def update_status(
        self,
        scene_id: str,
        status: str,
        patches_count: int = 0,
        error_msg: Optional[str] = None,
    ):
        """updates processing checkpoint status for a specific scene."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            cursor.execute(
                """
                UPDATE scenes_tracker
                SET status = ?, patches_count = ?, error_msg = ?, updated_at = ?
                WHERE scene_id = ?
                """,
                (status, patches_count, error_msg, now, scene_id),
            )
            conn.commit()

    def get_statistics(self) -> Dict[str, int]:
        """returns summary counts of dataset processing states."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, COUNT(*), SUM(patches_count) 
                FROM scenes_tracker 
                GROUP BY status
                """
            )
            stats = {}
            for row in cursor.fetchall():
                status, count, patches = row
                stats[f"scenes_{status.lower()}"] = count
                if patches:
                    stats[f"patches_{status.lower()}"] = patches
            return stats


class DatasetGenerator:
    """batch processor creating hdf5 splits from csv point records with state recovery."""

    def __init__(
        self,
        points_csv_path: str,
        output_dir: str = "data/processed",
        patch_size: int = 256,
        stride: int = 256,
        max_invalid_ratio: float = 0.20,
    ):
        self.points_csv_path = points_csv_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.db_path = os.path.join(output_dir, "dataset_checkpoint.db")
        self.tracker = DatasetProgressTracker(self.db_path)

        self.collector = SingleSceneCollector()
        self.extractor = PatchExtractor(
            patch_size=patch_size,
            stride=stride,
            max_invalid_ratio=max_invalid_ratio,
        )

        # distinct writers for train, validation, and test splits
        self.writers: Dict[str, HDF5Writer] = {
            "train": HDF5Writer(os.path.join(output_dir, "train.h5"), patch_size=patch_size),
            "val": HDF5Writer(os.path.join(output_dir, "val.h5"), patch_size=patch_size),
            "test": HDF5Writer(os.path.join(output_dir, "test.h5"), patch_size=patch_size),
        }

    def run(self, max_scenes: Optional[int] = None):
        """executes batch data acquisition, slicing, and atomic saving."""
        df_points = pd.read_csv(self.points_csv_path)
        self.tracker.sync_from_csv(df_points)

        pending_scenes = self.tracker.get_pending_scenes()
        total_pending = len(pending_scenes)
        print(f"found {total_pending} scenes pending processing.")

        if max_scenes:
            pending_scenes = pending_scenes[:max_scenes]

        for idx, scene in enumerate(pending_scenes, 1):
            scene_id = scene["scene_id"]
            lat = scene["lat"]
            lon = scene["lon"]
            target_date = scene["target_date"]
            is_fire = scene["is_fire"]
            split = scene["split"]

            print(f"\n[{idx}/{len(pending_scenes)}] processing {scene_id} | split={split} | fire={is_fire}")
            self.tracker.update_status(scene_id, status="PROCESSING")

            t_start = time.perf_counter()
            try:
                # 1. fetch full multimodal scene
                sample = self.collector.collect_sample(
                    lat=lat,
                    lon=lon,
                    target_date=target_date,
                    is_fire=is_fire,
                )

                if sample is None:
                    print(f"failed to retrieve valid scene data for {scene_id}")
                    self.tracker.update_status(
                        scene_id, status="FAILED", error_msg="collector returned none"
                    )
                    continue

                # 2. slice scene into valid patches
                patches_batch = list(self.extractor.extract_patches(sample, scene_id=scene_id))

                if not patches_batch:
                    print(f"no patches passed validation filter for {scene_id}")
                    self.tracker.update_status(
                        scene_id, status="COMPLETED", patches_count=0, error_msg="0 patches passed validation"
                    )
                    continue

                # 3. write patches to appropriate h5 split
                writer = self.writers.get(split, self.writers["train"])
                written_count = writer.write_patches_batch(patches_batch)

                elapsed = time.perf_counter() - t_start
                print(f"saved {written_count} patches in {elapsed:.2f}s to {split}.h5")

                self.tracker.update_status(
                    scene_id, status="COMPLETED", patches_count=written_count
                )

            except Exception as e:
                print(f"error processing scene {scene_id}: {str(e)}")
                self.tracker.update_status(
                    scene_id, status="FAILED", error_msg=str(e)
                )

        print("\n" + "=" * 50)
        print("batch processing finished. status summary:")
        print(self.tracker.get_statistics())
        print("=" * 50)


if __name__ == "__main__":
    csv_input = "data\interim\points_dataset_thinned.csv"

    if not os.path.exists(csv_input):
        os.makedirs(os.path.dirname(csv_input), exist_ok=True)
        demo_data = {
            "lat": [56.7264, 56.8000, 56.6500],
            "lon": [-111.3803, -111.4500, -111.2000],
            "target_date": ["2023-06-15", "2023-06-20", "2023-07-02"],
            "is_fire": [1, 1, 0],
            "split": ["train", "val", "test"],
        }
        pd.DataFrame(demo_data).to_csv(csv_input, index=False)
        print(f"created sample csv at {csv_input}")

    generator = DatasetGenerator(
        points_csv_path=csv_input,
        output_dir="data\processed",
        patch_size=256,
        stride=256,
        max_invalid_ratio=0.20,
    )

    generator.run()