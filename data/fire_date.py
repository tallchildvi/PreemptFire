import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dotenv import load_dotenv, dotenv_values
from sklearn.cluster import DBSCAN, KMeans
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()


class Date:
    def __init__(self, fire_date: date, country_code: str = "CAN"):
        self.fire_date_obj = fire_date
        self.fire_date = fire_date.strftime("%Y-%m-%d")
        self.country_code = country_code.upper()

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
        """yield available firms api keys from env"""
        config = dotenv_values()
        for env_name, env_value in config.items():
            if env_name.startswith("FIRMS_API_KEY") and env_value:
                yield env_value

    def _load_country_bounds(self):
        """load country geometry and calculate bounding box"""
        try:
            url = f"https://raw.githubusercontent.com/johan/world.geo.json/master/countries/{self.country_code}.geo.json"
            gdf = gpd.read_file(url)
            minx, miny, maxx, maxy = gdf.total_bounds

            self.geometry = gdf.unary_union
            self.bbox_str = f"{minx:.1f},{miny:.1f},{maxx:.1f},{maxy:.1f}"
            self.lat_range = (miny, maxy)
            self.lon_range = (minx, maxx)
        except Exception as e:
            print(f"error loading geojson for {self.country_code}: {e}")

    def is_within_country(self, lat: float, lon: float) -> bool:
        """check if coordinates fall inside country boundary"""
        if self.geometry is None:
            return True
        return self.geometry.contains(Point(lon, lat))

    def generate_fires(self):
        """fetch active fires from firms api using available keys"""
        load_dotenv()

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
        """fetch and cache validation window fires in 5-day chunks"""
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

    @staticmethod
    def _haversine_distance_km(lat1, lon1, lat2_array, lon2_array):
        """calculate geodetic distance in kilometers"""
        R = 6371.0
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2_array), np.radians(lon2_array)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    def _sample_spatially_diverse(self, bin_df: pd.DataFrame, n_needed: int = 5) -> pd.DataFrame:
        """sample representative points across 2d spatial clusters using kmeans"""
        if len(bin_df) <= n_needed:
            return bin_df

        coords = bin_df[['latitude', 'longitude']].values
        n_clusters = min(n_needed, len(bin_df))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        bin_df = bin_df.copy()
        bin_df['spatial_cluster'] = kmeans.fit_predict(coords)

        idx = bin_df.groupby('spatial_cluster')['frp'].idxmax()
        return bin_df.loc[idx].drop(columns=['spatial_cluster'])

    def filter_fires(self, epsilon: float = 0.012):
        """filter fires within country boundaries, cluster with dbscan, and stratify by frp"""
        if self.df_area.empty:
            return

        # 1. Filter points inside country before clustering
        self.df_area = self.df_area[
            self.df_area.apply(lambda r: self.is_within_country(r['latitude'], r['longitude']), axis=1)
        ].reset_index(drop=True)

        if self.df_area.empty:
            return

        # 2. DBSCAN clustering
        X = np.radians(self.df_area[['latitude', 'longitude']].values)
        dbscan = DBSCAN(eps=epsilon, metric='haversine')
        self.df_area['claster_id'] = dbscan.fit_predict(X)

        # Separate noise (-1) to prevent collapsing unclustered fires
        noise_mask = self.df_area['claster_id'] == -1
        if noise_mask.any():
            self.df_area.loc[noise_mask, 'claster_id'] = np.arange(100000, 100000 + noise_mask.sum())

        idx = self.df_area.groupby('claster_id')['bright_ti4'].idxmax()
        self.df_area_filtered = self.df_area.loc[idx].copy()

        # 3. FRP stratification
        self.df_area_filtered['frp_bin'] = pd.qcut(
            self.df_area_filtered['frp'], q=3, labels=['small', 'medium', 'large']
        )

        self.df_area_filtered = self.df_area_filtered.groupby('frp_bin', group_keys=False).apply(
            lambda bin_group: self._sample_spatially_diverse(bin_group, n_needed=5)
        )

        self.df_area_filtered = self.df_area_filtered[
            ['latitude', 'longitude', 'acq_date', 'acq_time', 'bright_ti4', 'frp']
        ].copy()
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
        """generate spatially shifted hard negatives verified against window fires"""
        if self.df_area_filtered.empty:
            return pd.DataFrame()

        df_window_fires = self._fetch_window_fires(buffer_days=buffer_days)
        negative_points = []

        for _, pos_row in self.df_area_filtered.iterrows():
            pos_lat = pos_row['latitude']
            pos_lon = pos_row['longitude']
            pos_time = pos_row['acq_time']

            valid_neg_found = False
            attempts = 0

            while not valid_neg_found and attempts < max_attempts:
                attempts += 1

                dist_km = np.random.uniform(min_shift_km, max_shift_km)
                angle_rad = np.random.uniform(0, 2 * np.pi)

                delta_lat = (dist_km * np.cos(angle_rad)) / 111.0
                delta_lon = (dist_km * np.sin(angle_rad)) / (111.0 * np.cos(np.radians(pos_lat)))

                neg_lat = pos_lat + delta_lat
                neg_lon = pos_lon + delta_lon

                if not self.is_within_country(neg_lat, neg_lon):
                    continue

                if not df_window_fires.empty:
                    fire_distances = self._haversine_distance_km(
                        neg_lat, neg_lon,
                        df_window_fires['latitude'].values,
                        df_window_fires['longitude'].values
                    )
                    if fire_distances.min() <= safe_radius_km:
                        continue

                all_current_negatives = negative_points.copy()
                if not self.df_negatives.empty:
                    all_current_negatives.extend(self.df_negatives[['latitude', 'longitude']].to_dict('records'))

                if all_current_negatives:
                    prev_neg_lats = np.array([p['latitude'] for p in all_current_negatives])
                    prev_neg_lons = np.array([p['longitude'] for p in all_current_negatives])
                    neg_distances = self._haversine_distance_km(neg_lat, neg_lon, prev_neg_lats, prev_neg_lons)
                    if neg_distances.min() <= min_neg_dist_km:
                        continue

                valid_neg_found = True

            if valid_neg_found:
                negative_points.append({
                    'latitude': round(neg_lat, 5),
                    'longitude': round(neg_lon, 5),
                    'acq_date': self.fire_date,
                    'acq_time': int(pos_time),
                    'bright_ti4': 0.0,
                    'frp': 0.0,
                    'is_fire': 0
                })

        if negative_points:
            df_new_neg = pd.DataFrame(negative_points)
            self.df_negatives = pd.concat([self.df_negatives, df_new_neg], ignore_index=True)

        return self.df_negatives

    def generate_random_negatives(
        self,
        n_points: int = 5,
        safe_radius_km: float = 25.0,
        min_neg_dist_km: float = 15.0,
        buffer_days: int = 15
    ) -> pd.DataFrame:
        """generate random background negatives across country landmass"""
        df_window_fires = self._fetch_window_fires(buffer_days=buffer_days)
        random_negatives = []

        default_time = int(self.df_area_filtered['acq_time'].median()) if not self.df_area_filtered.empty else 1900

        attempts = 0
        while len(random_negatives) < n_points and attempts < 150:
            attempts += 1
            rand_lat = np.random.uniform(*self.lat_range)
            rand_lon = np.random.uniform(*self.lon_range)

            if not self.is_within_country(rand_lat, rand_lon):
                continue

            if not df_window_fires.empty:
                distances = self._haversine_distance_km(
                    rand_lat, rand_lon,
                    df_window_fires['latitude'].values,
                    df_window_fires['longitude'].values
                )
                if distances.min() <= safe_radius_km:
                    continue

            all_current_negatives = random_negatives.copy()
            if not self.df_negatives.empty:
                all_current_negatives.extend(self.df_negatives[['latitude', 'longitude']].to_dict('records'))

            if all_current_negatives:
                prev_lats = np.array([p['latitude'] for p in all_current_negatives])
                prev_lons = np.array([p['longitude'] for p in all_current_negatives])
                neg_distances = self._haversine_distance_km(rand_lat, rand_lon, prev_lats, prev_lons)
                if neg_distances.min() <= min_neg_dist_km:
                    continue

            random_negatives.append({
                'latitude': round(rand_lat, 5),
                'longitude': round(rand_lon, 5),
                'acq_date': self.fire_date,
                'acq_time': default_time,
                'bright_ti4': 0.0,
                'frp': 0.0,
                'is_fire': 0
            })

        if random_negatives:
            df_new_rand = pd.DataFrame(random_negatives)
            self.df_negatives = pd.concat([self.df_negatives, df_new_rand], ignore_index=True)

        return self.df_negatives

    def get_combined_dataset(self) -> pd.DataFrame:
        """return combined dataset of positive and negative points"""
        if self.df_area_filtered.empty and self.df_negatives.empty:
            return pd.DataFrame()

        return pd.concat([self.df_area_filtered, self.df_negatives], ignore_index=True)

    def plot_static_scatter(self):
        """plot points on top of landmass map"""
        df_plot = self.get_combined_dataset()
        if df_plot.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        if self.geometry is not None:
            gpd.GeoSeries([self.geometry]).plot(
                ax=ax,
                color='#e2e8f0',
                edgecolor='#64748b',
                linewidth=0.8,
                alpha=0.9
            )

        sns.scatterplot(
            data=df_plot,
            x='longitude',
            y='latitude',
            hue='is_fire',
            style='is_fire',
            palette={1: '#e11d48', 0: '#16a34a'},
            markers={1: 'X', 0: 'o'},
            s=120,
            edgecolor='black',
            linewidth=0.8,
            ax=ax,
            zorder=3
        )

        plt.title(f"fires (1) vs negatives (0) on map of {self.country_code} ({self.fire_date})", fontsize=14, pad=12)
        plt.xlabel("longitude", fontsize=11)
        plt.ylabel("latitude", fontsize=11)

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=['hard/random negative (0)', 'active fire (1)'],
            loc='lower left',
            frameon=True,
            facecolor='white',
            framealpha=0.9
        )

        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_date = date(2023, 6, 15)
    date_obj = Date(test_date, country_code="CAN")

    date_obj.generate_fires()
    date_obj.filter_fires(epsilon=0.012)
    date_obj.generate_hard_negatives(buffer_days=15)
    date_obj.generate_random_negatives(n_points=5, buffer_days=15)

    df_day = date_obj.get_combined_dataset()
    print(df_day[['latitude', 'longitude', 'acq_date', 'acq_time', 'frp', 'is_fire']])
    date_obj.plot_static_scatter()