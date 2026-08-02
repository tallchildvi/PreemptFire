import os
import io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

class Date:
    def __init__(self, fire_date: date):
        self.fire_date_obj = fire_date
        self.fire_date = fire_date.strftime("%Y-%m-%d")
        self.df_area = pd.DataFrame()
        self.df_area_filtered = pd.DataFrame()
        self.df_negatives = pd.DataFrame()

    def generate_fires(self):
        """Fetch active fires from NASA FIRMS API for target date T_0."""
        load_dotenv()
        map_key = os.getenv("FIRMS_API_KEY")
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_SP/-141,41.5,-52.0,83.5/1/{self.fire_date}"

        try:
            self.df_area = pd.read_csv(url)
        except Exception as e:
            print(f"Error querying FIRMS API for {self.fire_date}: {e}")

    def _fetch_window_fires(self, buffer_days: int = 10) -> pd.DataFrame:
        """Fetch all fires within [T_0 - buffer_days, T_0 + buffer_days] in 3-day chunks."""
        load_dotenv()
        map_key = os.getenv("FIRMS_API_KEY")
        
        start_date = self.fire_date_obj - timedelta(days=buffer_days)
        end_date = self.fire_date_obj + timedelta(days=buffer_days)
        
        dfs = []
        curr_date = start_date
        
        # Fetch in 3-day chunks to stay within FIRMS SP query limits
        while curr_date <= end_date:
            days_to_fetch = min(3, (end_date - curr_date).days + 1)
            date_str = curr_date.strftime('%Y-%m-%d')
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_SP/-141,41.5,-52.0,83.5/{days_to_fetch}/{date_str}"
            
            try:
                df_chunk = pd.read_csv(url)
                if not df_chunk.empty and 'latitude' in df_chunk.columns:
                    dfs.append(df_chunk)
            except Exception as e:
                print(f"Error fetching validation chunk for {date_str}: {e}")
                
            curr_date += timedelta(days=days_to_fetch)
            
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    @staticmethod
    def _haversine_distance_km(lat1, lon1, lat2_array, lon2_array):
        """Calculate geodetic distance in km between a point and an array of coordinates."""
        R = 6371.0  
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2_array), np.radians(lon2_array)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    def _sample_south_biased_with_north(self, bin_df: pd.DataFrame, n_needed: int = 5) -> pd.DataFrame:
        """Sample points biased towards the south, ensuring at least 1 northernmost point."""
        if len(bin_df) <= n_needed:
            return bin_df
        
        sorted_df = bin_df.sort_values(by='latitude', ascending=True)
        n_south = max(1, n_needed - 1)
        south_points = sorted_df.iloc[:n_south]
        north_point = sorted_df.iloc[-1:]
        return pd.concat([south_points, north_point]).drop_duplicates()

    def filter_fires(self, epsilon: float):
        """Cluster fires using DBSCAN and sample stratified points by FRP."""
        if self.df_area.empty:
            print("No fires available to filter.")
            return

        X = np.radians(self.df_area[['latitude', 'longitude']].values)
        dbscan = DBSCAN(eps=epsilon, metric='haversine')
        self.df_area['claster_id'] = dbscan.fit_predict(X)
        
        idx = self.df_area.groupby('claster_id')['bright_ti4'].idxmax()
        self.df_area_filtered = self.df_area.loc[idx].copy()
        
        self.df_area_filtered['frp_bin'] = pd.qcut(
            self.df_area_filtered['frp'], q=3, labels=['small', 'medium', 'large']
        )

        self.df_area_filtered = self.df_area_filtered.groupby('frp_bin', group_keys=False).apply(
            lambda bin_group: self._sample_south_biased_with_north(bin_group, n_needed=5)
        )
        self.df_area_filtered = self.df_area_filtered[['latitude', 'longitude', 'acq_date', 'bright_ti4', 'frp']].copy()
        self.df_area_filtered['is_fire'] = 1

    def generate_hard_negatives(
        self, 
        min_shift_km: float = 30.0, 
        max_shift_km: float = 70.0, 
        safe_radius_km: float = 25.0, 
        min_neg_dist_km: float = 15.0,  # Minimum distance between negative points
        buffer_days: int = 10,
        max_attempts: int = 30
    ) -> pd.DataFrame:
        """Generate hard negative points (is_fire = 0) verified against window fires and existing negatives."""
        if self.df_area_filtered.empty:
            print("No positive fires available to generate negatives.")
            return pd.DataFrame()

        print(f"Fetching validation fires for +/-{buffer_days} days around {self.fire_date}...")
        df_window_fires = self._fetch_window_fires(buffer_days=buffer_days)

        negative_points = []

        for _, pos_row in self.df_area_filtered.iterrows():
            pos_lat = pos_row['latitude']
            pos_lon = pos_row['longitude']
            
            valid_neg_found = False
            attempts = 0

            while not valid_neg_found and attempts < max_attempts:
                attempts += 1

                # Random spatial shift
                dist_km = np.random.uniform(min_shift_km, max_shift_km)
                angle_rad = np.random.uniform(0, 2 * np.pi)

                delta_lat = (dist_km * np.cos(angle_rad)) / 111.0
                delta_lon = (dist_km * np.sin(angle_rad)) / (111.0 * np.cos(np.radians(pos_lat)))

                neg_lat = pos_lat + delta_lat
                neg_lon = pos_lon + delta_lon

                # 1. Check distance to real fires (+/- buffer_days)
                if not df_window_fires.empty:
                    fire_distances = self._haversine_distance_km(
                        neg_lat, neg_lon, 
                        df_window_fires['latitude'].values, 
                        df_window_fires['longitude'].values
                    )
                    if fire_distances.min() <= safe_radius_km:
                        continue  # Too close to an active or past/future fire

                # 2. Check distance to previously generated negative points
                if negative_points:
                    prev_neg_lats = np.array([p['latitude'] for p in negative_points])
                    prev_neg_lons = np.array([p['longitude'] for p in negative_points])
                    neg_distances = self._haversine_distance_km(
                        neg_lat, neg_lon, 
                        prev_neg_lats, 
                        prev_neg_lons
                    )
                    if neg_distances.min() <= min_neg_dist_km:
                        continue  # Too close to another negative point

                # Point passes both spatial checks
                valid_neg_found = True

            if valid_neg_found:
                negative_points.append({
                    'latitude': round(neg_lat, 5),
                    'longitude': round(neg_lon, 5),
                    'acq_date': self.fire_date,
                    'bright_ti4': 0.0,
                    'frp': 0.0,
                    'is_fire': 0
                })

        self.df_negatives = pd.DataFrame(negative_points)
        print(f"Generated {len(self.df_negatives)} sterile and isolated hard negative points.")
        return self.df_negatives

    def get_combined_dataset(self) -> pd.DataFrame:
        """Return combined dataset of positive and negative points."""
        if self.df_area_filtered.empty or self.df_negatives.empty:
            print("Warning: Missing positive or negative points.")
            
        return pd.concat([self.df_area_filtered, self.df_negatives], ignore_index=True)

    def plot_static_scatter(self):
        """Plot positive and negative points on a scatter map."""
        df_plot = self.get_combined_dataset()
        if df_plot.empty:
            print("No data available to plot.")
            return

        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(
            data=df_plot,
            x='longitude',
            y='latitude',
            hue='is_fire',
            style='is_fire',
            palette={1: 'red', 0: 'green'},
            markers={1: 'X', 0: 'o'},
            s=100,
            edgecolor='black'
        )

        plt.axhline(y=49.0, color='blue', linestyle='--', alpha=0.5, label='USA Border (~49°N)')
        plt.axhline(y=60.0, color='purple', linestyle='--', alpha=0.5, label='Northern Border (~60°N)')

        plt.title(f"Fire (1) vs Hard Negative (0) Points for {self.fire_date}", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_date = date(2023, 6, 15)
    date_obj = Date(test_date)
    
    # 1. Fetch fires for target date
    date_obj.generate_fires()
    
    # 2. Filter positive points
    date_obj.filter_fires(epsilon=0.012)
    
    # 3. Generate negatives validated against +/-10 days window
    date_obj.generate_hard_negatives(buffer_days=10)
    
    # 4. Get combined daily dataset
    df_day = date_obj.get_combined_dataset()
    print(df_day[['latitude', 'longitude', 'acq_date', 'frp', 'is_fire']])
    
    # 5. Plot distribution
    date_obj.plot_static_scatter()