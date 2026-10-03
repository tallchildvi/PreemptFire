from datetime import date, datetime, timedelta
import os
from pathlib import Path
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
from shapely.geometry import Point
from sklearn.cluster import DBSCAN, KMeans

from src.config import BASE_DIR

load_dotenv(BASE_DIR / ".env")

DEFAULT_INDUSTRIAL_HOTSPOTS_PATH = BASE_DIR / "data" / "raw" / "industrial_hotspots.csv"
DEFAULT_GLOBFIRE_PATH = BASE_DIR / "data" / "raw" / "globfire"
BOUNDARIES_CACHE_DIR = BASE_DIR / "data" / "raw" / "boundaries"


class Date:
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

        self.geometry = None
        self.bbox_str = "-180,-90,180,90"
        self.lat_range = (-90.0, 90.0)
        self.lon_range = (-180.0, 180.0)

        self._load_country_bounds()

    def _get_api_keys(self):
        """yield available firms api keys from env safely"""
        config = dotenv_values(BASE_DIR / ".env")
        for env_name, env_value in config.items():
            if env_name.startswith("FIRMS_API_KEY") and env_value:
                yield env_value

    def _load_country_bounds(self):
        """load country geometry, cache locally to avoid rate limits, and calculate bounding box"""
        BOUNDARIES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_path = BOUNDARIES_CACHE_DIR / f"{self.country_code}.geo.json"

        try:
            if not local_path.exists():
                url = f"https://raw.githubusercontent.com/johan/world.geo.json/master/countries/{self.country_code}.geo.json"
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

            gdf = gpd.read_file(local_path)
            minx, miny, maxx, maxy = gdf.total_bounds

            self.geometry = gdf.unary_union
            self.bbox_str = f"{minx:.1f},{miny:.1f},{maxx:.1f},{maxy:.1f}"
            self.lat_range = (miny, maxy)
            self.lon_range = (minx, maxx)
        except Exception as e:
            warnings.warn(f"Error loading geojson for {self.country_code}: {e}. Falling back to global bbox.")

    def _get_xyz(self, lat_series, lon_series) -> np.ndarray:
        """convert lat/lon to 3d spherical coordinates for exact cKDTree distance"""
        lat_rad, lon_rad = np.radians(lat_series), np.radians(lon_series)
        return np.column_stack([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ])

    def generate_fires(self):
        for key in self._get_api_keys():
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_SP/{self.bbox_str}/1/{self.fire_date}"
            try:
                df = pd.read_csv(url)
                if not df.empty and 'latitude' in df.columns:
                    self.df_area = df
                    return
            except Exception:
                continue
        print(f"failed to query firms api for {self.fire_date}")

    def _fetch_window_fires(self, buffer_days: int = 15) -> pd.DataFrame:
        if not self.df_window_fires.empty:
            return self.df_window_fires

        start_date = self.fire_date_obj - timedelta(days=buffer_days)
        end_date = self.fire_date_obj + timedelta(days=buffer_days)

        dfs = []
        curr_date = start_date

        while curr_date <= end_date:
            days_to_fetch = min(5, (end_date - curr_date).days + 1)
            date_str = curr_date.strftime('%Y-%m-%d')

            for key in self._get_api_keys():
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_SP/{self.bbox_str}/{days_to_fetch}/{date_str}"
                try:
                    df_chunk = pd.read_csv(url)
                    if not df_chunk.empty and 'latitude' in df_chunk.columns:
                        dfs.append(df_chunk)
                        break
                except Exception:
                    continue

            curr_date += timedelta(days=days_to_fetch)

        self.df_window_fires = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return self.df_window_fires

    def _filter_industrial_hotspots(self, industrial_radius_km: float = 2.0) -> None:
        if self.df_area.empty:
            return

        path = self.industrial_blacklist_path
        if path is None or not Path(path).exists():
            warnings.warn(f"Industrial blacklist not found at {path}. Skipping filtering.")
            return

        try:
            path = Path(path)
            if path.suffix.lower() == ".csv":
                ind_df = pd.read_csv(path)
            elif path.suffix.lower() in [".gpkg", ".shp", ".geojson"]:
                ind_gdf = gpd.read_file(path)
                ind_df = pd.DataFrame({"latitude": ind_gdf.geometry.y, "longitude": ind_gdf.geometry.x})
            else:
                return

            lat_col = "latitude" if "latitude" in ind_df.columns else "lat"
            lon_col = "longitude" if "longitude" in ind_df.columns else "lon"
            ind_df = ind_df.dropna(subset=[lat_col, lon_col])
            
            if ind_df.empty: return

            ind_xyz = self._get_xyz(ind_df[lat_col].values, ind_df[lon_col].values)
            tree = cKDTree(ind_xyz)
            pts_xyz = self._get_xyz(self.df_area['latitude'].values, self.df_area['longitude'].values)

            chord_dist = 2.0 * np.sin(industrial_radius_km / (2.0 * 6371.0))
            matches = tree.query_ball_point(pts_xyz, r=chord_dist)
            clean_mask = [len(m) == 0 for m in matches]

            dropped_count = len(self.df_area) - sum(clean_mask)
            if dropped_count > 0:
                print(f"  [Verification] Dropped {dropped_count} points near industrial heat sources (< {industrial_radius_km} km)")

            self.df_area = self.df_area[clean_mask].reset_index(drop=True)
        except Exception as e:
            warnings.warn(f"Error filtering industrial hotspots: {e}")

    def _verify_globfire_burned_areas(self, buffer_meters: float = 500.0, tolerance_days: int = 3) -> None:
        if self.df_area.empty:
            return

        path = self.globfire_path
        if path is None or not Path(path).exists():
            warnings.warn(f"GlobFire file missing at {path}. Skipping verification.")
            return

        path = Path(path)
        if path.is_dir():
            candidates = list(path.glob("*.gpkg")) + list(path.glob("*.shp")) + list(path.glob("*.geojson"))
            if not candidates:
                warnings.warn(f"No GlobFire vector files found in {path}. Skipping verification.")
                return
            path = candidates[0]

        try:
            min_lat, max_lat = self.df_area['latitude'].min() - 0.1, self.df_area['latitude'].max() + 0.1
            min_lon, max_lon = self.df_area['longitude'].min() - 0.1, self.df_area['longitude'].max() + 0.1

            try:
                gdf_globfire = gpd.read_file(path, bbox=(min_lon, min_lat, max_lon, max_lat))
            except Exception:
                gdf_globfire = gpd.read_file(path)

            if gdf_globfire.empty:
                warnings.warn(f"No GlobFire data in this bbox. Dropping all {len(self.df_area)} points.")
                self.df_area = self.df_area.iloc[0:0]
                return

            gdf_globfire = gdf_globfire.to_crs("EPSG:4326")
            
            cols_lower = {c.lower(): c for c in gdf_globfire.columns}
            init_col = next((cols_lower[k] for k in cols_lower if 'init' in k or 'start' in k), None)
            end_col = next((cols_lower[k] for k in cols_lower if 'fin' in k or 'end' in k or 'last' in k), None)

            if init_col and end_col:
                target_dt = pd.to_datetime(self.fire_date)
                init_dates = pd.to_datetime(gdf_globfire[init_col], errors='coerce')
                end_dates = pd.to_datetime(gdf_globfire[end_col], errors='coerce')

                t_min = target_dt - pd.Timedelta(days=tolerance_days)
                t_max = target_dt + pd.Timedelta(days=tolerance_days)
                time_mask = (init_dates <= t_max) & (end_dates >= t_min)
                gdf_globfire = gdf_globfire[time_mask]

                if gdf_globfire.empty:
                    print(f"  [Verification] GlobFire has no matching events for {self.fire_date}. Dropping all points.")
                    self.df_area = self.df_area.iloc[0:0]
                    return

            gdf_points = gpd.GeoDataFrame(
                self.df_area.copy(),
                geometry=gpd.points_from_xy(self.df_area['longitude'], self.df_area['latitude']),
                crs="EPSG:4326",
            )

            # Accurate dynamic UTM projection for correct buffering
            utm_crs = gdf_points.estimate_utm_crs()
            gdf_points_buffered = gdf_points.to_crs(utm_crs)
            gdf_points_buffered['geometry'] = gdf_points_buffered.geometry.buffer(buffer_meters)
            gdf_points_buffered = gdf_points_buffered.to_crs("EPSG:4326")

            joined = gpd.sjoin(gdf_points_buffered, gdf_globfire, how="inner", predicate="intersects")
            valid_indices = joined.index.unique()

            dropped_count = len(self.df_area) - len(valid_indices)
            if dropped_count > 0:
                print(f"  [Verification] Retained {len(valid_indices)} GlobFire-verified fires (dropped {dropped_count} unverified)")

            self.df_area = self.df_area.loc[self.df_area.index.isin(valid_indices)].reset_index(drop=True)
        except Exception as e:
            warnings.warn(f"Error verifying against GlobFire: {e}")

    def _sample_spatially_diverse(self, bin_df: pd.DataFrame, n_needed: int = 5) -> pd.DataFrame:
        """sample representative points across 2d spatial clusters using kmeans"""
        if len(bin_df) <= n_needed:
            return bin_df

        coords = bin_df[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=min(n_needed, len(bin_df)), random_state=42, n_init=10)
        bin_df = bin_df.copy()
        bin_df['spatial_cluster'] = kmeans.fit_predict(coords)

        idx = bin_df.groupby('spatial_cluster')['frp'].idxmax()
        return bin_df.loc[idx].drop(columns=['spatial_cluster'])

    def filter_fires(
        self,
        epsilon: float = 0.012, # intended to remain radians 
        lookback_days: int = 15,
        ignition_radius_km: float = 20.0,
        industrial_radius_km: float = 2.0,
        globfire_buffer_m: float = 500.0,
        globfire_tolerance_days: int = 3,
        verify_industrial: bool = True,
        verify_globfire: bool = True,
    ):
        if self.df_area.empty: return

        # 1. Vectorized Point-in-Country check
        pts = gpd.points_from_xy(self.df_area['longitude'], self.df_area['latitude'])
        mask = pts.within(self.geometry)
        self.df_area = self.df_area[mask].reset_index(drop=True)

        if self.df_area.empty: return

        if verify_industrial:
            self._filter_industrial_hotspots(industrial_radius_km=industrial_radius_km)

        if self.df_area.empty: return

        if verify_globfire:
            self._verify_globfire_burned_areas(buffer_meters=globfire_buffer_m, tolerance_days=globfire_tolerance_days)

        if self.df_area.empty: return

        X = np.radians(self.df_area[['latitude', 'longitude']].values)
        dbscan = DBSCAN(eps=epsilon, metric='haversine')
        self.df_area['cluster_id'] = dbscan.fit_predict(X)

        noise_mask = self.df_area['cluster_id'] == -1
        if noise_mask.any():
            self.df_area.loc[noise_mask, 'cluster_id'] = np.arange(100000, 100000 + noise_mask.sum())

        idx = self.df_area.groupby('cluster_id')['bright_ti4'].idxmax()
        self.df_area_filtered = self.df_area.loc[idx].copy()

        # 5. Vectorized Lookback Filter using cKDTree
        df_window_fires = self._fetch_window_fires(buffer_days=lookback_days)
        if not df_window_fires.empty:
            past_fires = df_window_fires[pd.to_datetime(df_window_fires['acq_date']) < pd.to_datetime(self.fire_date)]
            if not past_fires.empty:
                past_xyz = self._get_xyz(past_fires['latitude'].values, past_fires['longitude'].values)
                tree = cKDTree(past_xyz)
                curr_xyz = self._get_xyz(self.df_area_filtered['latitude'].values, self.df_area_filtered['longitude'].values)
                chord_dist = 2.0 * np.sin(ignition_radius_km / (2.0 * 6371.0))
                
                dists, _ = tree.query(curr_xyz, k=1)
                valid_mask = dists > chord_dist
                self.df_area_filtered = self.df_area_filtered[valid_mask].copy()

        if self.df_area_filtered.empty: return

        if len(self.df_area_filtered) > 5 and self.df_area_filtered['frp'].nunique() > 1:
            # fixed pandas 2.2 deprecation warning using include_groups=False
            self.df_area_filtered['frp_bin'] = pd.qcut(self.df_area_filtered['frp'], q=min(3, self.df_area_filtered['frp'].nunique()), duplicates="drop")
            self.df_area_filtered = self.df_area_filtered.groupby('frp_bin', group_keys=False, observed=False).apply(
                lambda b: self._sample_spatially_diverse(b, n_needed=5), include_groups=False
            )
        elif len(self.df_area_filtered) > 5:
            self.df_area_filtered = self._sample_spatially_diverse(self.df_area_filtered, n_needed=5)

        self.df_area_filtered = self.df_area_filtered[['latitude', 'longitude', 'acq_date', 'acq_time', 'bright_ti4', 'frp']].copy()
        self.df_area_filtered['is_fire'] = 1

    def generate_hard_negatives(
        self,
        min_shift_km: float = 30.0,
        max_shift_km: float = 70.0,
        safe_radius_km: float = 25.0,
        min_neg_dist_km: float = 15.0,
        buffer_days: int = 15,
        max_attempts: int = 30
    ) -> pd.DataFrame:
        if self.df_area_filtered.empty:
            return pd.DataFrame()

        df_window_fires = self._fetch_window_fires(buffer_days=buffer_days)
        fire_tree = None
        safe_chord = 2.0 * np.sin(safe_radius_km / (2.0 * 6371.0))
        if not df_window_fires.empty:
            fire_xyz = self._get_xyz(df_window_fires['latitude'].values, df_window_fires['longitude'].values)
            fire_tree = cKDTree(fire_xyz)

        neg_dist_chord = 2.0 * np.sin(min_neg_dist_km / (2.0 * 6371.0))
        valid_negatives = []

        for _, pos_row in self.df_area_filtered.iterrows():
            pos_lat, pos_lon = pos_row['latitude'], pos_row['longitude']
            
            dist_km = np.random.uniform(min_shift_km, max_shift_km, size=max_attempts)
            angle_rad = np.random.uniform(0, 2 * np.pi, size=max_attempts)
            delta_lat = (dist_km * np.cos(angle_rad)) / 111.0
            delta_lon = (dist_km * np.sin(angle_rad)) / (111.0 * np.cos(np.radians(pos_lat)))
            
            cand_lats, cand_lons = pos_lat + delta_lat, pos_lon + delta_lon

            pts = gpd.points_from_xy(cand_lons, cand_lats)
            mask_country = pts.within(self.geometry)
            cand_lats, cand_lons = cand_lats[mask_country], cand_lons[mask_country]

            if len(cand_lats) == 0: continue

            if fire_tree is not None:
                cand_xyz = self._get_xyz(cand_lats, cand_lons)
                dists, _ = fire_tree.query(cand_xyz, k=1)
                safe_mask = dists > safe_chord
                cand_lats, cand_lons = cand_lats[safe_mask], cand_lons[safe_mask]

            for cand_lat, cand_lon in zip(cand_lats, cand_lons):
                conflict = False
                cand_xyz = self._get_xyz([cand_lat], [cand_lon])[0]
                
                # Check against already added negatives to keep spread
                existing = valid_negatives + self.df_negatives.to_dict('records')
                if existing:
                    ex_xyz = self._get_xyz([e['latitude'] for e in existing], [e['longitude'] for e in existing])
                    tree = cKDTree(ex_xyz)
                    d, _ = tree.query(cand_xyz, k=1)
                    if d <= neg_dist_chord:
                        conflict = True
                
                if not conflict:
                    valid_negatives.append({
                        'latitude': round(cand_lat, 5), 'longitude': round(cand_lon, 5),
                        'acq_date': self.fire_date, 'acq_time': int(pos_row['acq_time']),
                        'bright_ti4': 0.0, 'frp': 0.0, 'is_fire': 0
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
        intensity: float = 0.0
    ) -> pd.DataFrame:
        df_window_fires = self._fetch_window_fires(buffer_days=buffer_days)
        default_time = int(self.df_area_filtered['acq_time'].median()) if not self.df_area_filtered.empty else 1900

        fire_tree = None
        safe_chord = 2.0 * np.sin(safe_radius_km / (2.0 * 6371.0))
        if not df_window_fires.empty:
            fire_xyz = self._get_xyz(df_window_fires['latitude'].values, df_window_fires['longitude'].values)
            fire_tree = cKDTree(fire_xyz)

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

            pts = gpd.points_from_xy(rand_lon, rand_lat)
            mask_country = pts.within(self.geometry)
            rand_lat, rand_lon = rand_lat[mask_country], rand_lon[mask_country]

            if len(rand_lat) == 0: continue

            if fire_tree is not None:
                cand_xyz = self._get_xyz(rand_lat, rand_lon)
                dists, _ = fire_tree.query(cand_xyz, k=1)
                safe_mask = dists > safe_chord
                rand_lat, rand_lon = rand_lat[safe_mask], rand_lon[safe_mask]

            for lat, lon in zip(rand_lat, rand_lon):
                if len(valid_negatives) >= n_points: break
                
                conflict = False
                cand_xyz = self._get_xyz([lat], [lon])[0]
                existing = valid_negatives + self.df_negatives.to_dict('records')
                if existing:
                    ex_xyz = self._get_xyz([e['latitude'] for e in existing], [e['longitude'] for e in existing])
                    tree = cKDTree(ex_xyz)
                    d, _ = tree.query(cand_xyz, k=1)
                    if d <= neg_dist_chord:
                        conflict = True

                if not conflict:
                    valid_negatives.append({
                        'latitude': round(lat, 5), 'longitude': round(lon, 5),
                        'acq_date': self.fire_date, 'acq_time': default_time,
                        'bright_ti4': 0.0, 'frp': 0.0, 'is_fire': 0
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
        if df_plot.empty: return

        fig, ax = plt.subplots(figsize=(12, 8))
        if self.geometry is not None:
            gpd.GeoSeries([self.geometry]).plot(ax=ax, color='#e2e8f0', edgecolor='#64748b', linewidth=0.8, alpha=0.9)

        sns.scatterplot(
            data=df_plot, x='longitude', y='latitude', hue='is_fire', style='is_fire',
            palette={1: '#e11d48', 0: '#16a34a'}, markers={1: 'X', 0: 'o'}, s=120,
            edgecolor='black', linewidth=0.8, ax=ax, zorder=3
        )
        
        plt.title(f"fires (1) vs negatives (0) on map of {self.country_code} ({self.fire_date})", fontsize=14, pad=12)
        plt.xlabel("longitude", fontsize=11)
        plt.ylabel("latitude", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_date = date(2023, 6, 15)
    date_obj = Date(test_date, country_code="CAN")
    date_obj.generate_fires()
    date_obj.filter_fires()
    date_obj.generate_hard_negatives()
    date_obj.generate_random_negatives(n_points=5, direction_north=1.0, direction_east=0.3, intensity=0.8)
    print(date_obj.get_combined_dataset())