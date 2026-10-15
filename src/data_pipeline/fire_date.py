from datetime import date, timedelta
from pathlib import Path
import random
import threading
from typing import Optional
import warnings
import requests

from dotenv import dotenv_values, load_dotenv
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans

from src.config import BASE_DIR

load_dotenv(BASE_DIR / ".env")

DEFAULT_INDUSTRIAL_HOTSPOTS_PATH = BASE_DIR / "data" / "raw" / "industrial_hotspots.csv"
DEFAULT_GLOBFIRE_PATH = BASE_DIR / "data" / "raw" / "globfire"
BOUNDARIES_CACHE_DIR = BASE_DIR / "data" / "raw" / "boundaries"
FIRMS_CACHE_DIR = BASE_DIR / "data" / "raw" / "firms_cache"


class Date:
    # class-level caches shared across threads to prevent redundant disk/network i/o
    _industrial_tree_cache: dict[str, Optional[cKDTree]] = {}
    _boundary_cache: dict[str, tuple] = {}
    _globfire_gdf_cache: dict[str, gpd.GeoDataFrame] = {}
    _cache_lock = threading.Lock()
    _chunk_locks: dict[str, threading.Lock] = {}

    def __init__(
        self,
        fire_date: date,
        country_code: str = "CAN",
        industrial_blacklist_path: Optional[Path | str] = None,
        globfire_path: Optional[Path | str] = None,
    ):
        self.fire_date_obj = fire_date
        self.fire_date = fire_date.strftime("%Y-%m-%d")
        self.country_code = country_code.upper()

        self.industrial_blacklist_path = (
            Path(industrial_blacklist_path)
            if industrial_blacklist_path is not None
            else DEFAULT_INDUSTRIAL_HOTSPOTS_PATH
        )
        self.globfire_path = (
            Path(globfire_path)
            if globfire_path is not None
            else DEFAULT_GLOBFIRE_PATH
        )

        self.df_area = pd.DataFrame()
        self.df_area_filtered = pd.DataFrame()
        self.df_negatives = pd.DataFrame()
        self.df_window_fires = pd.DataFrame()
        self._points_prefiltered = False

        self.geometry = None
        self.bbox_str = "-180,-90,180,90"
        self.lat_range = (-90.0, 90.0)
        self.lon_range = (-180.0, 180.0)

        self._load_country_bounds()

    def _get_api_keys(self):
        """yield available firms api keys in randomized order to balance rate limits across threads"""
        config = dotenv_values(BASE_DIR / ".env")
        keys = [
            env_value
            for env_name, env_value in config.items()
            if env_name.startswith("FIRMS_API_KEY") and env_value
        ]
        random.shuffle(keys)
        for key in keys:
            yield key

    def _load_country_bounds(self):
        """load country geometry, cache locally and in memory, and calculate bounding box"""
        with self._cache_lock:
            if self.country_code in self._boundary_cache:
                self.geometry, self.bbox_str, self.lat_range, self.lon_range = self._boundary_cache[self.country_code]
                return

        BOUNDARIES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_path = BOUNDARIES_CACHE_DIR / f"{self.country_code}.geo.json"

        try:
            with self._cache_lock:
                if not local_path.exists():
                    url = f"https://raw.githubusercontent.com/johan/world.geo.json/master/countries/{self.country_code}.geo.json"
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(response.text)

            gdf = gpd.read_file(local_path)
            minx, miny, maxx, maxy = gdf.total_bounds

            self.geometry = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
            self.bbox_str = f"{minx:.1f},{miny:.1f},{maxx:.1f},{maxy:.1f}"
            self.lat_range = (miny, maxy)
            self.lon_range = (minx, maxx)

            with self._cache_lock:
                self._boundary_cache[self.country_code] = (
                    self.geometry,
                    self.bbox_str,
                    self.lat_range,
                    self.lon_range,
                )
        except Exception as e:
            warnings.warn(f"error loading geojson for {self.country_code}: {e}. falling back to global bbox.")

    @staticmethod
    def _get_xyz(lat_series, lon_series) -> np.ndarray:
        """convert lat/lon to 3d spherical coordinates for exact ckdtree distance"""
        lat_rad, lon_rad = np.radians(lat_series), np.radians(lon_series)
        return np.column_stack([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ])

    # =========================================================================
    # FIRMS 10-DAY CHUNK DISK CACHE (MINIMIZES API CALLS)
    # =========================================================================

    def _get_chunk_lock(self, chunk_key: str) -> threading.Lock:
        with self._cache_lock:
            if chunk_key not in self._chunk_locks:
                self._chunk_locks[chunk_key] = threading.Lock()
            return self._chunk_locks[chunk_key]

    def _fetch_10d_block(self, block_start: date, days_count: int) -> pd.DataFrame:
        """fetch a single <=10-day firms block and cache it on disk for reuse across overlapping windows"""
        FIRMS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        start_str = block_start.strftime("%Y-%m-%d")
        cache_file = FIRMS_CACHE_DIR / f"{self.country_code}_{start_str}_{days_count}d.csv"
        chunk_lock = self._get_chunk_lock(str(cache_file))

        with chunk_lock:
            if cache_file.exists():
                try:
                    return pd.read_csv(cache_file)
                except pd.errors.EmptyDataError:
                    return pd.DataFrame()
                except Exception:
                    pass

            for key in self._get_api_keys():
                url = (
                    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                    f"{key}/VIIRS_SNPP_SP/{self.bbox_str}/{days_count}/{start_str}"
                )
                try:
                    df_chunk = pd.read_csv(url)
                    if "latitude" in df_chunk.columns:
                        tmp_file = cache_file.with_suffix(".tmp")
                        df_chunk.to_csv(tmp_file, index=False)
                        tmp_file.replace(cache_file)
                        return df_chunk
                except Exception:
                    continue

            print(f"failed to query firms api block {start_str} ({days_count}d) for {self.country_code}")
            return pd.DataFrame()

    def _fetch_firms_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """cover any [start_date, end_date] interval using deterministic 10-day cached blocks"""
        blocks = []
        curr = start_date
        while curr <= end_date:
            # align block start to deterministic 10-day calendar boundaries within each year
            doy_zero = (curr - date(curr.year, 1, 1)).days
            block_start = date(curr.year, 1, 1) + timedelta(days=(doy_zero // 10) * 10)
            year_end = date(curr.year, 12, 31)
            block_end = min(year_end, block_start + timedelta(days=9))
            days_count = (block_end - block_start).days + 1

            blocks.append((block_start, days_count))
            curr = block_end + timedelta(days=1)

        dfs = []
        for b_start, b_days in blocks:
            df_b = self._fetch_10d_block(b_start, b_days)
            if not df_b.empty and "acq_date" in df_b.columns:
                dfs.append(df_b)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")
        mask = (combined["acq_date"] >= s_str) & (combined["acq_date"] <= e_str)
        return combined[mask].reset_index(drop=True)

    # =========================================================================
    # STAGE 1: EARLY POINT-LEVEL FILTERING & POLYGON PRE-CHECK
    # =========================================================================

    def _get_industrial_tree(self) -> Optional[cKDTree]:
        """load and cache ckdtree of industrial/agricultural hotspots"""
        path = self.industrial_blacklist_path
        cache_key = str(path.resolve()) if path is not None else "none"

        with self._cache_lock:
            if cache_key in self._industrial_tree_cache:
                return self._industrial_tree_cache[cache_key]

        if path is None or not Path(path).exists():
            warnings.warn(f"industrial blacklist not found at {path}. skipping filtering.")
            with self._cache_lock:
                self._industrial_tree_cache[cache_key] = None
            return None

        try:
            path = Path(path)
            if path.suffix.lower() == ".csv":
                ind_df = pd.read_csv(path)
            elif path.suffix.lower() in [".gpkg", ".shp", ".geojson"]:
                ind_gdf = gpd.read_file(path)
                ind_df = pd.DataFrame({"latitude": ind_gdf.geometry.y, "longitude": ind_gdf.geometry.x})
            else:
                with self._cache_lock:
                    self._industrial_tree_cache[cache_key] = None
                return None

            lat_col = "latitude" if "latitude" in ind_df.columns else "lat"
            lon_col = "longitude" if "longitude" in ind_df.columns else "lon"
            ind_df = ind_df.dropna(subset=[lat_col, lon_col])

            if ind_df.empty:
                with self._cache_lock:
                    self._industrial_tree_cache[cache_key] = None
                return None

            ind_xyz = self._get_xyz(ind_df[lat_col].values, ind_df[lon_col].values)
            tree = cKDTree(ind_xyz)
            with self._cache_lock:
                self._industrial_tree_cache[cache_key] = tree
            return tree
        except Exception as e:
            warnings.warn(f"error loading industrial hotspots: {e}")
            with self._cache_lock:
                self._industrial_tree_cache[cache_key] = None
            return None

    def _resolve_globfire_file(self) -> Optional[Path]:
        """find the burned area vector file matching the target year"""
        path = self.globfire_path
        if path is None or not Path(path).exists():
            warnings.warn(f"globfire path missing at {path}. skipping verification.")
            return None

        path = Path(path)
        if path.is_file():
            return path

        candidates = (
            list(path.glob("*.geojson"))
            + list(path.glob("*.gpkg"))
            + list(path.glob("*.shp"))
        )
        if not candidates:
            warnings.warn(f"no globfire/effis vector files found in {path}. skipping verification.")
            return None

        target_year = str(self.fire_date_obj.year)
        year_matches = [f for f in candidates if target_year in f.stem]
        if year_matches:
            return year_matches[0]

        warnings.warn(f"no burned area file specifically for year {target_year} in {path}. skipping verification.")
        return None

    def _load_year_burned_polygons(self) -> Optional[gpd.GeoDataFrame]:
        """load and cache the entire year's burned area polygons in memory"""
        vector_file = self._resolve_globfire_file()
        if vector_file is None:
            return None

        cache_key = str(vector_file.resolve())
        with self._cache_lock:
            if cache_key in self._globfire_gdf_cache:
                return self._globfire_gdf_cache[cache_key]

        try:
            gdf = gpd.read_file(vector_file)
            if not gdf.empty:
                gdf = gdf.to_crs("EPSG:4326")
                cols_lower = {c.lower(): c for c in gdf.columns}
                init_col = next((cols_lower[k] for k in cols_lower if "init" in k or "start" in k or "firedate" in k), None)
                end_col = next((cols_lower[k] for k in cols_lower if "fin" in k or "end" in k or "last" in k), None)

                if init_col and end_col:
                    gdf["_init_dt"] = pd.to_datetime(gdf[init_col], errors="coerce")
                    gdf["_end_dt"] = pd.to_datetime(gdf[end_col], errors="coerce")

            with self._cache_lock:
                self._globfire_gdf_cache[cache_key] = gdf
            return gdf
        except Exception as e:
            warnings.warn(f"error reading burned area file {vector_file}: {e}")
            return None

    def _get_active_burned_polygons(self, tolerance_days: int = 3) -> Optional[gpd.GeoDataFrame]:
        """
        return burned area polygons active on self.fire_date within the country bbox.
        returns None if verification is unavailable, or an empty/non-empty GeoDataFrame if available.
        """
        gdf_year = self._load_year_burned_polygons()
        if gdf_year is None:
            return None
        if gdf_year.empty:
            return gdf_year

        gdf_sub = gdf_year
        if "_init_dt" in gdf_sub.columns and "_end_dt" in gdf_sub.columns:
            target_dt = pd.to_datetime(self.fire_date)
            t_min = target_dt - pd.Timedelta(days=tolerance_days)
            t_max = target_dt + pd.Timedelta(days=tolerance_days)
            time_mask = (gdf_sub["_init_dt"] <= t_max) & (gdf_sub["_end_dt"] >= t_min)
            gdf_sub = gdf_sub[time_mask]

        if gdf_sub.empty:
            return gdf_sub

        # filter by country bounding box with a small margin
        min_lat, max_lat = self.lat_range[0] - 0.1, self.lat_range[1] + 0.1
        min_lon, max_lon = self.lon_range[0] - 0.1, self.lon_range[1] + 0.1
        gdf_sub = gdf_sub.cx[min_lon:max_lon, min_lat:max_lat]
        return gdf_sub

    def _filter_df_by_country_and_blacklist(
        self,
        df: pd.DataFrame,
        industrial_radius_km: float = 2.0,
        verify_industrial: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """early filtering: drop low-confidence sensor noise, out-of-country points, and industrial zones"""
        if df.empty:
            return df

        if "confidence" in df.columns:
            conf_str = df["confidence"].astype(str).str.lower()
            df = df[conf_str != "l"].reset_index(drop=True)
            if df.empty:
                return df

        if self.geometry is not None:
            pts = gpd.points_from_xy(df["longitude"], df["latitude"])
            mask = pts.within(self.geometry)
            df = df[mask].reset_index(drop=True)
            if df.empty:
                return df

        if verify_industrial:
            tree = self._get_industrial_tree()
            if tree is not None:
                pts_xyz = self._get_xyz(df["latitude"].values, df["longitude"].values)
                chord_dist = 2.0 * np.sin(industrial_radius_km / (2.0 * 6371.0))
                matches = tree.query_ball_point(pts_xyz, r=chord_dist)
                clean_mask = np.array([len(m) == 0 for m in matches], dtype=bool)

                dropped_count = len(df) - int(clean_mask.sum())
                if verbose and dropped_count > 0:
                    print(f"  [early filter] dropped {dropped_count} points near industrial/agro sources (< {industrial_radius_km} km)")

                df = df[clean_mask].reset_index(drop=True)

        return df

    def _verify_df_against_burned_areas(
        self,
        df: pd.DataFrame,
        buffer_meters: float = 500.0,
        tolerance_days: int = 3,
    ) -> pd.DataFrame:
        """early verification: retain only points intersecting confirmed burned area polygons"""
        if df.empty:
            return df

        gdf_burned = self._get_active_burned_polygons(tolerance_days=tolerance_days)
        if gdf_burned is None:
            return df

        if gdf_burned.empty:
            print(f"  [early filter] no matching burned area events for {self.fire_date}. dropping all points.")
            return df.iloc[0:0].copy()

        try:
            gdf_points = gpd.GeoDataFrame(
                df.copy(),
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326",
            )

            utm_crs = gdf_points.estimate_utm_crs()
            gdf_points_buffered = gdf_points.to_crs(utm_crs)
            gdf_points_buffered["geometry"] = gdf_points_buffered.geometry.buffer(buffer_meters)
            gdf_points_buffered = gdf_points_buffered.to_crs("EPSG:4326")

            joined = gpd.sjoin(gdf_points_buffered, gdf_burned, how="inner", predicate="intersects")
            valid_indices = joined.index.unique()

            dropped_count = len(df) - len(valid_indices)
            if dropped_count > 0:
                print(f"  [early filter] retained {len(valid_indices)} burned-area verified points (dropped {dropped_count})")

            return df.loc[df.index.isin(valid_indices)].reset_index(drop=True)
        except Exception as e:
            warnings.warn(f"error verifying against burned areas: {e}")
            return df

    def prefilter_points(
        self,
        industrial_radius_km: float = 2.0,
        globfire_buffer_m: float = 500.0,
        globfire_tolerance_days: int = 3,
        verify_industrial: bool = True,
        verify_globfire: bool = True,
    ) -> pd.DataFrame:
        """run all early point-level filters on self.df_area before centroid aggregation"""
        if self.df_area.empty:
            self._points_prefiltered = True
            return self.df_area

        self.df_area = self._filter_df_by_country_and_blacklist(
            self.df_area,
            industrial_radius_km=industrial_radius_km,
            verify_industrial=verify_industrial,
            verbose=True,
        )

        if not self.df_area.empty and verify_globfire:
            self.df_area = self._verify_df_against_burned_areas(
                self.df_area,
                buffer_meters=globfire_buffer_m,
                tolerance_days=globfire_tolerance_days,
            )

        self._points_prefiltered = True
        return self.df_area

    def generate_fires(
        self,
        early_filter: bool = True,
        industrial_radius_km: float = 2.0,
        globfire_buffer_m: float = 500.0,
        globfire_tolerance_days: int = 3,
        verify_industrial: bool = True,
        verify_globfire: bool = True,
    ):
        """
        check if local effis polygons exist for this date first;
        if yes, load/fetch 10-day cached firms block and pre-filter points.
        """
        self._points_prefiltered = False

        # polygon-first pre-check: skip nasa firms api completely if effis has 0 active fires on this date
        if early_filter and verify_globfire:
            active_polys = self._get_active_burned_polygons(tolerance_days=globfire_tolerance_days)
            if active_polys is not None and active_polys.empty:
                self.df_area = pd.DataFrame()
                self._points_prefiltered = True
                return

        self.df_area = self._fetch_firms_range(self.fire_date_obj, self.fire_date_obj)
        if self.df_area.empty:
            return

        if early_filter:
            self.prefilter_points(
                industrial_radius_km=industrial_radius_km,
                globfire_buffer_m=globfire_buffer_m,
                globfire_tolerance_days=globfire_tolerance_days,
                verify_industrial=verify_industrial,
                verify_globfire=verify_globfire,
            )

    def _fetch_window_fires(
        self,
        buffer_days: int = 15,
        industrial_radius_km: float = 2.0,
        verify_industrial: bool = True,
    ) -> pd.DataFrame:
        """fetch +/- buffer_days window using 10-day cached blocks and clean out industrial hotspots"""
        if not self.df_window_fires.empty:
            return self.df_window_fires

        start_date = self.fire_date_obj - timedelta(days=buffer_days)
        end_date = self.fire_date_obj + timedelta(days=buffer_days)

        raw_window = self._fetch_firms_range(start_date, end_date)
        if raw_window.empty:
            self.df_window_fires = pd.DataFrame()
            return self.df_window_fires

        self.df_window_fires = self._filter_df_by_country_and_blacklist(
            raw_window,
            industrial_radius_km=industrial_radius_km,
            verify_industrial=verify_industrial,
            verbose=False,
        )
        return self.df_window_fires

    # =========================================================================
    # STAGE 2: CENTROID AGGREGATION & FIRE-LEVEL FILTERING (LOOKBACK)
    # =========================================================================

    def _aggregate_to_centroids(self, cluster_radius_km: float = 15.0) -> pd.DataFrame:
        """merge nearby verified fire pixels belonging to the same fire into a single centroid"""
        if self.df_area.empty:
            return pd.DataFrame()

        eps_rad = cluster_radius_km / 6371.0
        coords_rad = np.radians(self.df_area[["latitude", "longitude"]].values)

        dbscan = DBSCAN(eps=eps_rad, metric="haversine", min_samples=1)
        df_clustered = self.df_area.copy()
        df_clustered["cluster_id"] = dbscan.fit_predict(coords_rad)

        centroids = (
            df_clustered.groupby("cluster_id", as_index=False)
            .agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                acq_date=("acq_date", "first"),
                acq_time=("acq_time", "min"),
                bright_ti4=("bright_ti4", "max"),
                frp=("frp", "max"),
            )
        )
        centroids["latitude"] = centroids["latitude"].round(5)
        centroids["longitude"] = centroids["longitude"].round(5)
        return centroids.drop(columns=["cluster_id"])

    def _sample_spatially_diverse(self, bin_df: pd.DataFrame, n_needed: int = 5) -> pd.DataFrame:
        """optional helper: sample representative points across 2d spatial clusters using kmeans"""
        if len(bin_df) <= n_needed:
            return bin_df

        coords = bin_df[["latitude", "longitude"]].values
        kmeans = KMeans(n_clusters=min(n_needed, len(bin_df)), random_state=42, n_init=10)
        bin_df = bin_df.copy()
        bin_df["spatial_cluster"] = kmeans.fit_predict(coords)

        idx = bin_df.groupby("spatial_cluster")["frp"].idxmax()
        return bin_df.loc[idx].drop(columns=["spatial_cluster"])

    def filter_fires(
        self,
        cluster_radius_km: float = 3.0,
        epsilon: Optional[float] = None,
        lookback_days: int = 15,
        ignition_radius_km: float = 20.0,
        industrial_radius_km: float = 2.0,
        globfire_buffer_m: float = 500.0,
        globfire_tolerance_days: int = 3,
        verify_industrial: bool = True,
        verify_globfire: bool = True,
        max_fires_per_bin: Optional[int] = None,
    ):
        """
        stage 1: ensure raw points are filtered (country, blacklist, burned area).
        stage 2: merge pixels of the same fire into centroids.
        stage 3: filter out ongoing fires using 15-day lookback window.
        """
        if self.df_area.empty:
            return

        if not self._points_prefiltered:
            self.prefilter_points(
                industrial_radius_km=industrial_radius_km,
                globfire_buffer_m=globfire_buffer_m,
                globfire_tolerance_days=globfire_tolerance_days,
                verify_industrial=verify_industrial,
                verify_globfire=verify_globfire,
            )

        if self.df_area.empty:
            return

        effective_radius_km = cluster_radius_km
        if epsilon is not None and epsilon < 0.005:
            effective_radius_km = epsilon * 6371.0

        self.df_area_filtered = self._aggregate_to_centroids(cluster_radius_km=effective_radius_km)

        if self.df_area_filtered.empty:
            return

        df_window_fires = self._fetch_window_fires(
            buffer_days=lookback_days,
            industrial_radius_km=industrial_radius_km,
            verify_industrial=verify_industrial,
        )
        if not df_window_fires.empty:
            past_fires = df_window_fires[
                pd.to_datetime(df_window_fires["acq_date"]) < pd.to_datetime(self.fire_date)
            ]
            if not past_fires.empty:
                past_xyz = self._get_xyz(past_fires["latitude"].values, past_fires["longitude"].values)
                tree = cKDTree(past_xyz)
                curr_xyz = self._get_xyz(
                    self.df_area_filtered["latitude"].values,
                    self.df_area_filtered["longitude"].values,
                )
                chord_dist = 2.0 * np.sin(ignition_radius_km / (2.0 * 6371.0))

                dists, _ = tree.query(curr_xyz, k=1)
                valid_mask = dists > chord_dist
                self.df_area_filtered = self.df_area_filtered[valid_mask].reset_index(drop=True)

        if self.df_area_filtered.empty:
            return

        if max_fires_per_bin is not None:
            if len(self.df_area_filtered) > max_fires_per_bin and self.df_area_filtered["frp"].nunique() > 1:
                self.df_area_filtered["frp_bin"] = pd.qcut(
                    self.df_area_filtered["frp"],
                    q=min(3, self.df_area_filtered["frp"].nunique()),
                    duplicates="drop",
                )
                sampled_bins = [
                    self._sample_spatially_diverse(group.drop(columns=["frp_bin"]), n_needed=max_fires_per_bin)
                    for _, group in self.df_area_filtered.groupby("frp_bin", observed=False)
                    if not group.empty
                ]
                self.df_area_filtered = pd.concat(sampled_bins, ignore_index=True)
            elif len(self.df_area_filtered) > max_fires_per_bin:
                self.df_area_filtered = self._sample_spatially_diverse(
                    self.df_area_filtered, n_needed=max_fires_per_bin
                )

        self.df_area_filtered = self.df_area_filtered[
            ["latitude", "longitude", "acq_date", "acq_time", "bright_ti4", "frp"]
        ].copy()
        self.df_area_filtered["is_fire"] = 1

    # =========================================================================
    # STAGE 3: NEGATIVE SAMPLING
    # =========================================================================

    def generate_hard_negatives(
        self,
        min_shift_km: float = 30.0,
        max_shift_km: float = 70.0,
        safe_radius_km: float = 25.0,
        min_neg_dist_km: float = 15.0,
        buffer_days: int = 15,
        max_attempts: int = 30,
    ) -> pd.DataFrame:
        if self.df_area_filtered.empty:
            return pd.DataFrame()

        df_window_fires = self._fetch_window_fires(buffer_days=buffer_days)
        fire_tree = None
        safe_chord = 2.0 * np.sin(safe_radius_km / (2.0 * 6371.0))
        if not df_window_fires.empty:
            fire_xyz = self._get_xyz(df_window_fires["latitude"].values, df_window_fires["longitude"].values)
            fire_tree = cKDTree(fire_xyz)

        ind_tree = self._get_industrial_tree()
        ind_chord = 2.0 * np.sin(2.0 / (2.0 * 6371.0))

        neg_dist_chord = 2.0 * np.sin(min_neg_dist_km / (2.0 * 6371.0))
        valid_negatives = []

        for _, pos_row in self.df_area_filtered.iterrows():
            pos_lat, pos_lon = pos_row["latitude"], pos_row["longitude"]

            dist_km = np.random.uniform(min_shift_km, max_shift_km, size=max_attempts)
            angle_rad = np.random.uniform(0, 2 * np.pi, size=max_attempts)
            delta_lat = (dist_km * np.cos(angle_rad)) / 111.0
            delta_lon = (dist_km * np.sin(angle_rad)) / (111.0 * np.cos(np.radians(pos_lat)))

            cand_lats, cand_lons = pos_lat + delta_lat, pos_lon + delta_lon

            if self.geometry is not None:
                pts = gpd.points_from_xy(cand_lons, cand_lats)
                mask_country = pts.within(self.geometry)
                cand_lats, cand_lons = cand_lats[mask_country], cand_lons[mask_country]

            if len(cand_lats) == 0:
                continue

            if ind_tree is not None:
                cand_xyz = self._get_xyz(cand_lats, cand_lons)
                matches = ind_tree.query_ball_point(cand_xyz, r=ind_chord)
                ind_mask = np.array([len(m) == 0 for m in matches], dtype=bool)
                cand_lats, cand_lons = cand_lats[ind_mask], cand_lons[ind_mask]

            if len(cand_lats) == 0:
                continue

            if fire_tree is not None:
                cand_xyz = self._get_xyz(cand_lats, cand_lons)
                dists, _ = fire_tree.query(cand_xyz, k=1)
                safe_mask = dists > safe_chord
                cand_lats, cand_lons = cand_lats[safe_mask], cand_lons[safe_mask]

            for cand_lat, cand_lon in zip(cand_lats, cand_lons):
                conflict = False
                cand_xyz = self._get_xyz([cand_lat], [cand_lon])[0]

                existing = valid_negatives + self.df_negatives.to_dict("records")
                if existing:
                    ex_xyz = self._get_xyz([e["latitude"] for e in existing], [e["longitude"] for e in existing])
                    tree = cKDTree(ex_xyz)
                    d, _ = tree.query(cand_xyz, k=1)
                    if d <= neg_dist_chord:
                        conflict = True

                if not conflict:
                    valid_negatives.append({
                        "latitude": round(cand_lat, 5),
                        "longitude": round(cand_lon, 5),
                        "acq_date": self.fire_date,
                        "acq_time": int(pos_row["acq_time"]),
                        "bright_ti4": 0.0,
                        "frp": 0.0,
                        "is_fire": 0,
                    })
                    break

        if valid_negatives:
            self.df_negatives = pd.concat([self.df_negatives, pd.DataFrame(valid_negatives)], ignore_index=True)
        return self.df_negatives

    def generate_random_negatives(
        self,
        n_points: int = 5,
        safe_radius_km: float = 25.0,
        min_neg_dist_km: float = 15.0,
        buffer_days: int = 15,
        direction_north: float = 0.0,
        direction_east: float = 0.0,
        intensity: float = 0.0,
    ) -> pd.DataFrame:
        df_window_fires = self._fetch_window_fires(buffer_days=buffer_days)
        default_time = int(self.df_area_filtered["acq_time"].median()) if not self.df_area_filtered.empty else 1900

        fire_tree = None
        safe_chord = 2.0 * np.sin(safe_radius_km / (2.0 * 6371.0))
        if not df_window_fires.empty:
            fire_xyz = self._get_xyz(df_window_fires["latitude"].values, df_window_fires["longitude"].values)
            fire_tree = cKDTree(fire_xyz)

        ind_tree = self._get_industrial_tree()
        ind_chord = 2.0 * np.sin(2.0 / (2.0 * 6371.0))

        neg_dist_chord = 2.0 * np.sin(min_neg_dist_km / (2.0 * 6371.0))
        valid_negatives = []

        def biased_sample_vec(min_val, max_val, weight, intens, size):
            if intens <= 0.0 or weight == 0.0:
                return np.random.uniform(min_val, max_val, size)
            u = np.random.uniform(0, 1, size)
            power = 1.0 + (intens * abs(weight) * 49.0)
            u_biased = u ** (1.0 / power) if weight > 0 else u ** power
            return min_val + u_biased * (max_val - min_val)

        attempts = 0
        batch_size = max(100, n_points * 20)

        while len(valid_negatives) < n_points and attempts < 10:
            attempts += 1
            rand_lat = biased_sample_vec(self.lat_range[0], self.lat_range[1], direction_north, intensity, batch_size)
            rand_lon = biased_sample_vec(self.lon_range[0], self.lon_range[1], direction_east, intensity, batch_size)

            if self.geometry is not None:
                pts = gpd.points_from_xy(rand_lon, rand_lat)
                mask_country = pts.within(self.geometry)
                rand_lat, rand_lon = rand_lat[mask_country], rand_lon[mask_country]

            if len(rand_lat) == 0:
                continue

            if ind_tree is not None:
                cand_xyz = self._get_xyz(rand_lat, rand_lon)
                matches = ind_tree.query_ball_point(cand_xyz, r=ind_chord)
                ind_mask = np.array([len(m) == 0 for m in matches], dtype=bool)
                rand_lat, rand_lon = rand_lat[ind_mask], rand_lon[ind_mask]

            if len(rand_lat) == 0:
                continue

            if fire_tree is not None:
                cand_xyz = self._get_xyz(rand_lat, rand_lon)
                dists, _ = fire_tree.query(cand_xyz, k=1)
                safe_mask = dists > safe_chord
                rand_lat, rand_lon = rand_lat[safe_mask], rand_lon[safe_mask]

            for lat, lon in zip(rand_lat, rand_lon):
                if len(valid_negatives) >= n_points:
                    break

                conflict = False
                cand_xyz = self._get_xyz([lat], [lon])[0]
                existing = valid_negatives + self.df_negatives.to_dict("records")
                if existing:
                    ex_xyz = self._get_xyz([e["latitude"] for e in existing], [e["longitude"] for e in existing])
                    tree = cKDTree(ex_xyz)
                    d, _ = tree.query(cand_xyz, k=1)
                    if d <= neg_dist_chord:
                        conflict = True

                if not conflict:
                    valid_negatives.append({
                        "latitude": round(lat, 5),
                        "longitude": round(lon, 5),
                        "acq_date": self.fire_date,
                        "acq_time": default_time,
                        "bright_ti4": 0.0,
                        "frp": 0.0,
                        "is_fire": 0,
                    })

        if valid_negatives:
            self.df_negatives = pd.concat([self.df_negatives, pd.DataFrame(valid_negatives)], ignore_index=True)
        return self.df_negatives

    def get_combined_dataset(self) -> pd.DataFrame:
        if self.df_area_filtered.empty and self.df_negatives.empty:
            return pd.DataFrame()
        return pd.concat([self.df_area_filtered, self.df_negatives], ignore_index=True)

    def plot_static_scatter(self):
        df_plot = self.get_combined_dataset()
        if df_plot.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        if self.geometry is not None:
            gpd.GeoSeries([self.geometry]).plot(
                ax=ax, color="#e2e8f0", edgecolor="#64748b", linewidth=0.8, alpha=0.9
            )

        sns.scatterplot(
            data=df_plot,
            x="longitude",
            y="latitude",
            hue="is_fire",
            style="is_fire",
            palette={1: "#e11d48", 0: "#16a34a"},
            markers={1: "X", 0: "o"},
            s=120,
            edgecolor="black",
            linewidth=0.8,
            ax=ax,
            zorder=3,
        )

        plt.title(f"fires (1) vs negatives (0) on map of {self.country_code} ({self.fire_date})", fontsize=14, pad=12)
        plt.xlabel("longitude", fontsize=11)
        plt.ylabel("latitude", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_date = date(2023, 7, 15)
    date_obj = Date(test_date, country_code="ESP")
    date_obj.generate_fires()
    date_obj.filter_fires(cluster_radius_km=3.0)
    date_obj.generate_hard_negatives()
    date_obj.generate_random_negatives(n_points=5, direction_north=1.0, direction_east=0.3, intensity=0.8)
    print(date_obj.get_combined_dataset())