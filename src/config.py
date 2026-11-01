from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_H5_DIR = DATA_DIR / "processed"

# Spatial resolution & grid configuration: 12.80 km x 12.80 km at 100m/pixel (1 scene = 1 patch)
GRID_SIZE_PX = 128
PATCH_SIZE_PX = 128
STRIDE_PX = 128
PIXEL_SCALE_METERS = 100.0

HALF_SCENE_METERS = (GRID_SIZE_PX // 2) * PIXEL_SCALE_METERS  # 6400.0 meters

# Hard quality gate thresholds
MAX_INVALID_RATIO = 0.20
MIN_DEM_VALID_RATIO = 0.98