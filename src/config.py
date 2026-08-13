from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_H5_DIR = DATA_DIR / "processed"

GRID_SIZE_PX = 2048          
PATCH_SIZE_PX = 256          
PIXEL_SCALE_METERS = 10.0   

HALF_SCENE_METERS = (GRID_SIZE_PX // 2) * PIXEL_SCALE_METERS  