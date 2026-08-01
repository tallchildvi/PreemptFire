import pandas as pd
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, date
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
MAP_KEY = os.getenv("FIRMS_API_KEY")

class Date():
    def __init__(self, fire_date: date):
        self.fire_date = fire_date.strftime("%Y-%m-%d")
        self.df_area = pd.DataFrame()
        self.df_area_filtered = pd.DataFrame()

    def generate_fires(self):
        load_dotenv()
        MAP_KEY = os.getenv("FIRMS_API_KEY")
        url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/" + MAP_KEY + "/VIIRS_SNPP_SP/-141,41.5,-52.0,83.5/1/" + self.fire_date

        try:
            self.df_area = pd.read_csv(url)
            # print(self.df_area)

        except Exception as e:
            print(e)
            print ("There is an issue with the query. \nTry in your browser: %s" % url)

    def _sample_south_biased_with_north(self, bin_df: pd.DataFrame, n_needed: int = 5) -> pd.DataFrame:
        """
        Selects n_needed points: most from the south (closer to the US),
        but guaranteed to capture 1 point from the north.
        """
        if len(bin_df) <= n_needed:
            return bin_df
        
        sorted_df = bin_df.sort_values(by='latitude', ascending=True)
        n_south = max(1, n_needed - 1)
        south_points = sorted_df.iloc[:n_south]
        north_point = sorted_df.iloc[-1:]
        return pd.concat([south_points, north_point]).drop_duplicates()

    def filter_fires(self, epsilon: float):
        X = np.radians(self.df_area[['latitude', 'longitude']].values)
        dbscan = DBSCAN(eps=epsilon, metric='haversine')
        self.df_area['claster_id'] = dbscan.fit_predict(X)
        idx = self.df_area.groupby('claster_id')['bright_ti4'].idxmax()
        self.df_area_filtered = self.df_area.loc[idx]
        self.df_area_filtered['frp_bin'] = pd.qcut(self.df_area_filtered['frp'], q=3, labels=['small', 'medium', 'large'])


        self.df_area_filtered = self.df_area_filtered.groupby('frp_bin', group_keys=False).apply(
            lambda bin_group: self._sample_south_biased_with_north(bin_group, n_needed=6)
        )
        self.df_area_filtered = self.df_area_filtered[['latitude', 'longitude', 'acq_date', 'bright_ti4', 'frp']]

    def plot_static_scatter(self):
        if self.df_area_filtered.empty:
            print("No data")
            return

        plt.figure(figsize=(10, 6))
        
        scatter = sns.scatterplot(
            data=self.df_area_filtered,
            x='longitude',
            y='latitude',
            hue='frp',
            size='frp',
            sizes=(50, 300),
            palette='YlOrRd',
            edgecolor='black'
        )

        plt.axhline(y=49.0, color='blue', linestyle='--', alpha=0.5, label='Border with the USA(~49°N)')
        plt.axhline(y=60.0, color='purple', linestyle='--', alpha=0.5, label='Northern border (~60°N)')

        plt.title(f"Distribution of 15 selected fires by {self.fire_date}", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_date = date(2023, 7, 15)
    date1 = Date(test_date)
    date1.generate_fires()
    date1.filter_fires(0.012)
    print(date1.df_area_filtered)
    date1.plot_static_scatter()
    
