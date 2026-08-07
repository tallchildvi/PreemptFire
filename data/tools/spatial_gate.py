import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN

# calculate haversine distance matrix in kilometers
def haversine_matrix_km(coords: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon
    a = np.sin(dlat / 2.0)**2 + np.cos(lat)[:, None] * np.cos(lat) * np.sin(dlon / 2.0)**2
    return 2 * r * np.arcsin(np.sqrt(a))

# calculate global moran's i using standard pysal / esda implementation
def calculate_global_moran_i(df: pd.DataFrame, target_col: str = 'is_fire', k: int = 5) -> dict:
    n = len(df)
    if n <= k:
        return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0, "status": "TOO_FEW_POINTS"}

    # extract coordinates and target values
    coords = df[['longitude', 'latitude']].values
    y = df[target_col].values.astype(float)
    
    # build k-nearest neighbors spatial weights matrix
    w = KNN.from_array(coords, k=k)
    w.transform = 'R'  # row-standardization
    
    # compute global moran's i statistic
    mi = Moran(y, w)
    
    status = "PASSED" if abs(mi.I) <= 0.10 else "SPATIALLY_CORRELATED"
    
    return {
        "morans_i": round(float(mi.I), 4),
        "expected_i": round(float(mi.EI), 4),
        "z_score": round(float(mi.z_norm), 4),
        "p_value": "< 0.001" if mi.p_norm < 0.001 else round(float(mi.p_norm), 4),
        "status": status
    }

# perform spatial thinning to enforce minimum distance threshold
def spatial_thinning(df: pd.DataFrame, min_dist_km: float = 15.0, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    coords = df[['latitude', 'longitude']].values
    dist_matrix = haversine_matrix_km(coords)
    
    n = len(df)
    disabled = np.zeros(n, dtype=bool)
    selected_indices = []
    
    # shuffle indices to avoid spatial sequence bias
    permuted_indices = np.random.permutation(n)
    
    for idx in permuted_indices:
        if not disabled[idx]:
            selected_indices.append(idx)
            # mask out points within distance threshold
            disabled |= (dist_matrix[idx] < min_dist_km)
            
    return df.iloc[selected_indices].copy().reset_index(drop=True)

if __name__ == "__main__":
    # load input dataset and drop missing coordinate rows
    df = pd.read_csv("master_points_dataset.csv").dropna(subset=['latitude', 'longitude', 'is_fire'])
    
    # calculate initial spatial autocorrelation
    initial_stats = calculate_global_moran_i(df, target_col='is_fire', k=5)
    print("initial moran stats:", initial_stats)
    
    # apply spatial thinning filter
    thinned_df = spatial_thinning(df, min_dist_km=15.0)
    
    # calculate post-thinning spatial autocorrelation
    thinned_stats = calculate_global_moran_i(thinned_df, target_col='is_fire', k=5)
    print("thinned moran stats:", thinned_stats)
    
    # save thinned output dataset
    thinned_df.to_csv("master_points_thinned.csv", index=False)