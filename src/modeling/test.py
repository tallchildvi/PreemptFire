# import dataset

# ifuckinghatemyfuckinglife = dataset.WildfireDataset("data\processed\dataset.h5")
# print(ifuckinghatemyfuckinglife.channel_map)
# import pandas as pd
# df = pd.read_parquet("data/processed/tabular_features.parquet")

# # Перевіримо, чи там взагалі є ненульові значення:
# zero_cols = [
#     "int1_t0_to_target_temp_max", "cffdrs_fwi", "cffdrs_ffmc", 
#     "int2_tprev_to_t0_vpd_mean", "days_since_rain"
# ]
# print(df[zero_cols].describe().T[["mean", "std", "min", "max"]])
import json
from catboost import CatBoostClassifier

# Якщо модель збережена у файлі:
model = CatBoostClassifier()
model.load_model("catboost_baseline.cbm")

# Отримуємо всі параметри у вигляді словника
all_params = model.get_all_params()

# Гарний вивід у консоль через JSON:
print(json.dumps(all_params, indent=4, ensure_ascii=False))