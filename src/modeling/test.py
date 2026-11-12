import json
from catboost import CatBoostClassifier

# Load baseline CatBoost model
model = CatBoostClassifier()
model.load_model("catboost_baseline.cbm")

# Print model hyperparameters
all_params = model.get_all_params()
print(json.dumps(all_params, indent=4))