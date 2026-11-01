import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"

PF_MODE = os.getenv("PF_MODE", "100m")

if PF_MODE == "100m":
    GRID_SIZE_PX = 128
    PIXEL_SCALE_METERS = 100.0
    PATCH_SIZE_PX = 128
    PROCESSED_H5_DIR = DATA_DIR / "processed_100m"
else:
    GRID_SIZE_PX = 1280
    PIXEL_SCALE_METERS = 10.0
    PATCH_SIZE_PX = 256
    PROCESSED_H5_DIR = DATA_DIR / "processed"

HALF_SCENE_METERS = (GRID_SIZE_PX // 2) * PIXEL_SCALE_METERS   # 6400 м (100m) або 6400 м (10m): сцена 12.8 км
COARSE = PIXEL_SCALE_METERS >= 50.0