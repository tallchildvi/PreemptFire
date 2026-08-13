import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR


def spatiotemporal_thinning(
    df: pd.DataFrame, 
    min_dist_km: float = 40.0, 
    min_days: int = 15, 
    random_state: int = 42
) -> pd.DataFrame:
    """Greedy spatiotemporal thinning based on distance (km) and time (days) thresholds"""
    np.random.seed(random_state)
    df = df.copy().reset_index(drop=True)
    
    coords_rad = np.radians(df[['latitude', 'longitude']].values)
    dates = pd.to_datetime(df['acq_date']).values
    
    n = len(df)
    disabled = np.zeros(n, dtype=bool)
    selected_indices = []
    
    # random permutation to prevent systematic ordering bias
    permuted_indices = np.random.permutation(n)
    
    for idx in permuted_indices:
        if disabled[idx]:
            continue
            
        selected_indices.append(idx)
        
        # 1. calculate spatial distance using haversine
        dlat = coords_rad[:, 0] - coords_rad[idx, 0]
        dlon = coords_rad[:, 1] - coords_rad[idx, 1]
        a = (np.sin(dlat / 2.0)**2 + 
             np.cos(coords_rad[idx, 0]) * np.cos(coords_rad[:, 0]) * np.sin(dlon / 2.0)**2)
        spatial_dists = 2.0 * 6371.0 * np.arcsin(np.sqrt(a))
        
        # 2. calculate time difference in days
        time_diffs = np.abs((dates - dates[idx]).astype('timedelta64[D]').astype(int))
        
        # disable points falling within both spatial and temporal thresholds
        conflict_mask = (spatial_dists < min_dist_km) & (time_diffs <= min_days)
        disabled |= conflict_mask

    return df.iloc[selected_indices].copy().reset_index(drop=True)


def find_best_moran_subset(
    df: pd.DataFrame, 
    target_n: int = 1000, 
    n_iterations: int = 300, 
    k_neighbors: int = 5
) -> pd.DataFrame:
    """Find subset of target_n points minimizing absolute moran's i"""
    if len(df) < target_n:
        raise ValueError(f"not enough points after thinning ({len(df)} < {target_n}), reduce thinning thresholds")
        
    best_df = None
    best_moran_abs = float('inf')
    best_stats = {}

    print(f"Optimizing subset from {len(df)} to {target_n} points...")

    for i in range(n_iterations):
        sample_df = df.sample(n=target_n, random_state=i).copy()
        
        w = KNN.from_array(sample_df[['latitude', 'longitude']].values, k=k_neighbors)
        w.transform = 'R'
        
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

    print(f"Optimal subset found: moran's i = {best_stats['moran_i']:.5f} (p = {best_stats['p_value']:.3f})")
    return best_df


if __name__ == "__main__":
    input_file = RAW_DATA_DIR / "master_points_dataset.csv"
    output_file = INTERIM_DATA_DIR / "points_dataset_thinned.csv"
    df_raw = pd.read_csv(input_file)
    print(f"Initial points count: {len(df_raw)}")
    
    # spatiotemporal thinning (15 km / 15 days)
    df_thinned = spatiotemporal_thinning(df_raw, min_dist_km=20.0, min_days=15)
    print(f"points after spatiotemporal thinning: {len(df_thinned)}")
    
    # optimize subset to target_n points minimizing moran's i
    df_final = find_best_moran_subset(df_thinned, target_n=1000, n_iterations=500)
    df_final.to_csv(output_file, index=False)