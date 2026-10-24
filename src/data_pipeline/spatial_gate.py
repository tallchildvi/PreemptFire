import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from libpysal.weights import KNN
from esda.moran import Moran
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR


def _get_xyz(lat_array: np.ndarray, lon_array: np.ndarray) -> np.ndarray:
    """convert lat/lon arrays to 3d unit-sphere coordinates for fast ckdtree queries"""
    lat_rad = np.radians(lat_array)
    lon_rad = np.radians(lon_array)
    return np.column_stack([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])


def spatiotemporal_thinning(
    df: pd.DataFrame,
    min_dist_km: float = 20.0,
    min_days: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    chronological-priority spatiotemporal thinning:
    1. process confirmed fires (is_fire == 1) in strict chronological order (earliest date/time first)
       so day-0 ignitions are always kept and subsequent days of the same burn are suppressed.
    2. process negatives (is_fire == 0) in random order afterwards.
    """
    if df.empty:
        return df

    rng = np.random.default_rng(random_state)
    df = df.copy().reset_index(drop=True)

    xyz = _get_xyz(df["latitude"].values, df["longitude"].values)
    tree = cKDTree(xyz)
    chord_dist = 2.0 * np.sin(min_dist_km / (2.0 * 6371.0))

    dates = pd.to_datetime(df["acq_date"]).values.astype("datetime64[D]")

    # sort fires chronologically (earliest date & earliest acq_time first) to lock in day-0 ignitions
    fire_df = df[df["is_fire"] == 1].sort_values(by=["acq_date", "acq_time"], ascending=[True, True])
    fire_indices = fire_df.index.values

    # shuffle negatives randomly to avoid spatial ordering bias
    neg_indices = rng.permutation(df.index[df["is_fire"] == 0].values)

    ordered_indices = np.concatenate([fire_indices, neg_indices])

    n = len(df)
    disabled = np.zeros(n, dtype=bool)
    selected_indices = []

    for idx in ordered_indices:
        if disabled[idx]:
            continue

        selected_indices.append(idx)

        # query only spatial neighbors within min_dist_km using ckdtree
        neighbor_indices = tree.query_ball_point(xyz[idx], r=chord_dist)
        neighbor_indices = np.asarray(neighbor_indices, dtype=int)

        # check temporal difference only for spatial neighbors
        time_diffs = np.abs((dates[neighbor_indices] - dates[idx]).astype(int))
        conflict_neighbors = neighbor_indices[time_diffs <= min_days]
        disabled[conflict_neighbors] = True

    thinned_df = df.iloc[selected_indices].copy().reset_index(drop=True)
    return thinned_df


def find_best_moran_subset(
    df: pd.DataFrame,
    target_n: int = 1000,
    n_iterations: int = 500,
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """
    find strictly 50/50 class-balanced subset (equal fires and negatives)
    minimizing absolute moran's i spatial autocorrelation.
    """
    if df.empty:
        return df

    fires_df = df[df["is_fire"] == 1]
    negs_df = df[df["is_fire"] == 0]
    print(f"thinned pool composition: {len(fires_df)} fires (1), {len(negs_df)} negatives (0)")

    # enforce strict 50/50 balance based on the smaller class pool
    n_per_class = min(len(fires_df), len(negs_df), target_n // 2)
    if n_per_class == 0:
        print("error: one of the classes has 0 points after thinning")
        return pd.DataFrame()

    total_target = n_per_class * 2
    print(
        f"targeting strict 50/50 balance: {total_target} points total "
        f"({n_per_class} fires, {n_per_class} negatives)..."
    )

    # if pool sizes match n_per_class exactly, there is only one valid combination
    if len(fires_df) == n_per_class and len(negs_df) == n_per_class:
        sample_df = pd.concat([fires_df, negs_df], ignore_index=True)
        w = KNN.from_array(sample_df[["latitude", "longitude"]].values, k=min(k_neighbors, total_target - 1))
        w.transform = "R"
        mi = Moran(sample_df["is_fire"].values, w)
        print(f"exact pool match: moran's i = {mi.I:.5f} (p = {mi.p_sim:.3f})")
        return sample_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    best_df = None
    best_moran_abs = float("inf")
    best_stats = {}

    for i in range(n_iterations):
        # sample an equal count from each class on every iteration
        sampled_fires = fires_df.sample(n=n_per_class, random_state=i)
        sampled_negs = negs_df.sample(n=n_per_class, random_state=i + 10000)
        sample_df = pd.concat([sampled_fires, sampled_negs], ignore_index=True)

        w = KNN.from_array(sample_df[["latitude", "longitude"]].values, k=k_neighbors)
        w.transform = "R"

        y = sample_df["is_fire"].values
        mi = Moran(y, w)

        if abs(mi.I) < best_moran_abs:
            best_moran_abs = abs(mi.I)
            best_df = sample_df
            best_stats = {
                "moran_i": mi.I,
                "p_value": mi.p_sim,
                "z_score": mi.z_sim,
            }

    print(f"optimal subset found: moran's i = {best_stats['moran_i']:.5f} (p = {best_stats['p_value']:.3f})")
    return best_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    input_file = RAW_DATA_DIR / "master_points_dataset.csv"
    output_file = INTERIM_DATA_DIR / "points_dataset_thinned.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_file)
    print(f"initial points count: {len(df_raw)}")

    # spatiotemporal thinning (20 km / 15 days) with chronological day-0 fire priority
    df_thinned = spatiotemporal_thinning(df_raw, min_dist_km=20.0, min_days=15)
    print(f"points after spatiotemporal thinning: {len(df_thinned)}")

    # optimize subset minimizing moran's i with strict 50/50 class balance
    df_final = find_best_moran_subset(df_thinned, target_n=500, n_iterations=500)
    df_final.to_csv(output_file, index=False)
    print(f"saved thinned dataset to: {output_file}")