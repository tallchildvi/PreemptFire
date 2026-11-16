from pathlib import Path
import json
from catboost import CatBoostClassifier
from src.config import BASE_DIR, MODELS_DIR

model_candidates = [
    MODELS_DIR / "catboost_baseline.cbm",
    BASE_DIR / "catboost_baseline.cbm",
    Path("models/catboost_baseline.cbm"),
    Path("catboost_baseline.cbm"),
]
model_path = next((p for p in model_candidates if p.exists()), None)
if model_path is None:
    raise FileNotFoundError("catboost_baseline.cbm not found in models/ or project root.")

# Load baseline CatBoost model
model = CatBoostClassifier()
model.load_model(str(model_path))

# Print model hyperparameters
all_params = model.get_all_params()
print(json.dumps(all_params, indent=4))