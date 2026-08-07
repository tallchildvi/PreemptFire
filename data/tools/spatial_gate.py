import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from libpysal.weights import KNN
from esda.moran import Moran

def spatial_thinning(df, min_dist_km=45.0):
    """Жадібне розрідження точок за мінімальною відстанню"""
    coords = df[['latitude', 'longitude']].values
    # Переведення градусів у км (апроксимація для Haversine/Euclidean)
    coords_rad = np.radians(coords)
    
    keep_indices = []
    available = np.ones(len(df), dtype=bool)
    
    for i in range(len(df)):
        if not available[i]:
            continue
        keep_indices.append(i)
        
        # Обчислення відстані Haversine до всіх інших точок (в км)
        dlat = coords_rad[:, 0] - coords_rad[i, 0]
        dlon = coords_rad[:, 1] - coords_rad[i, 1]
        a = np.sin(dlat / 2)**2 + np.cos(coords_rad[i, 0]) * np.cos(coords_rad[:, 0]) * np.sin(dlon / 2)**2
        distances = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        
        # Виключаємо сусідів ближчих за min_dist_km
        available[distances < min_dist_km] = False

    return df.iloc[keep_indices].copy()

def find_best_moran_subset(df, target_n=1000, n_iterations=300, k_neighbors=5):
    """Пошук підмножини target_n точок з мінімальним Moran's I"""
    if len(df) < target_n:
        raise ValueError(f"Замало точок після розрідження ({len(df)} < {target_n})")
        
    best_df = None
    best_moran_abs = float('inf')
    best_stats = {}

    print(f"Оптимізація підмножини з {len(df)} точок до {target_n}...")

    for i in range(n_iterations):
        sample_df = df.sample(n=target_n, random_state=i).copy()
        
        # Обчислення Moran's I для вибірки
        w = KNN.from_array(sample_df[['latitude', 'longitude']].values, k=k_neighbors)
        w.transform = 'R'
        
        # Аналізуємо атрибут target або frp
        y = sample_df['is_fire'].values
        mi = Moran(y, w)
        
        if abs(mi.I) < best_moran_abs:
            best_moran_abs = abs(mi.I)
            best_df = sample_df
            best_stats = {
                'moran_i': mi.I,
                'p_value': mi.p_sim,
                'z_score': mi.z_sim
            }

    print(f"Знайдено оптимум: Moran's I = {best_stats['moran_i']:.5f} (p = {best_stats['p_value']:.3f})")
    return best_df





if __name__ == "__main__":
    df_raw = pd.read_csv("master_points_dataset.csv")
    df_thinned = spatial_thinning(df_raw, min_dist_km=40.0)
    df_final = find_best_moran_subset(df_thinned, target_n=1000, n_iterations=500)
    df_final.to_csv("points_dataset_thinned.csv", index=False)