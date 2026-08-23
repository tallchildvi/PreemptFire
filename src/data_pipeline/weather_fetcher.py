from datetime import datetime
from typing import Dict
import numpy as np
import requests


class WeatherFetcher:
    """
    Fetches historical reanalysis weather data via Open-Meteo Archive API.
    Provides decoupled methods for multi-day aggregate intervals and target day extremes.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def _fetch_hourly_data(self, lat: float, lon: float, start_date: str, end_date: str, variables: list) -> dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "UTC",
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            raise ValueError(f"No hourly weather data returned for {lat}, {lon} ({start_date} to {end_date})")

        return data["hourly"]

    def fetch_interval_metrics(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, float]:
        """
        Computes aggregate metrics for an arbitrary date range.
        Cumulative variables are normalized by the number of days in the interval.
        """
        variables = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
        hourly = self._fetch_hourly_data(lat, lon, start_date, end_date, variables)

        temps = np.array(hourly["temperature_2m"], dtype=np.float32)
        rhs = np.array(hourly["relative_humidity_2m"], dtype=np.float32)
        precips = np.array(hourly["precipitation"], dtype=np.float32)
        winds = np.array(hourly["wind_speed_10m"], dtype=np.float32)

        d_start = datetime.strptime(start_date, "%Y-%m-%d")
        d_end = datetime.strptime(end_date, "%Y-%m-%d")
        num_days = max(1, (d_end - d_start).days + 1)

        return {
            "temperature_mean": float(np.nanmean(temps)),
            "temperature_max": float(np.nanmax(temps)),
            "humidity_mean": float(np.nanmean(rhs)),
            "precip_daily_rate": float(np.nansum(precips) / num_days),
            "wind_speed_mean": float(np.nanmean(winds)),
        }

    def fetch_target_day_metrics(self, lat: float, lon: float, target_date: str) -> Dict[str, float]:
        """
        Computes extreme and mean meteorological metrics for the exact ignition/prediction target day.
        """
        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "precipitation",
        ]
        hourly = self._fetch_hourly_data(lat, lon, target_date, target_date, variables)

        temps = np.array(hourly["temperature_2m"], dtype=np.float32)
        rhs = np.array(hourly["relative_humidity_2m"], dtype=np.float32)
        dews = np.array(hourly["dew_point_2m"], dtype=np.float32)
        winds = np.array(hourly["wind_speed_10m"], dtype=np.float32)
        wind_dirs = np.array(hourly["wind_direction_10m"], dtype=np.float32)
        pressures = np.array(hourly["surface_pressure"], dtype=np.float32)
        precips = np.array(hourly["precipitation"], dtype=np.float32)

        # Vector mean for meteorological wind direction
        rads = np.radians(wind_dirs)
        mean_sin = np.nanmean(np.sin(rads))
        mean_cos = np.nanmean(np.cos(rads))
        mean_wind_dir = (np.degrees(np.arctan2(mean_sin, mean_cos)) + 360.0) % 360.0

        return {
            "temperature_2m_max": float(np.nanmax(temps)),
            "temperature_2m_min": float(np.nanmin(temps)),
            "relative_humidity_2m_min": float(np.nanmin(rhs)),
            "relative_humidity_2m_mean": float(np.nanmean(rhs)),
            "dew_point_2m_mean": float(np.nanmean(dews)),
            "wind_speed_10m_max": float(np.nanmax(winds)),
            "wind_direction_10m": float(mean_wind_dir),
            "surface_pressure_mean": float(np.nanmean(pressures)),
            "precipitation_sum": float(np.nansum(precips)),
        }


if __name__ == "__main__":
    lat, lon = 53.5461, -113.4937  # Edmonton

    # Scenario: 14 days between optical satellite passes, fire occurs on target_date
    date_t_prev = "2026-08-01"
    date_t0 = "2026-08-10"
    date_target = "2026-08-23"

    fetcher = WeatherFetcher()

    # 1. Period between T_prev and T_0 
    p2_metrics = fetcher.fetch_interval_metrics(lat, lon, start_date=date_t_prev, end_date=date_t0)

    # 2. Period before T_prev with the same 14-day duration
    p1_metrics = fetcher.fetch_interval_metrics(lat, lon, start_date="2023-06-02", end_date="2023-06-15")

    # 3. Target day conditions (fire ignition day)
    target_metrics = fetcher.fetch_target_day_metrics(lat, lon, target_date=date_target)

    print("\n--- Interval Weather [T_prev -> T_0] ---")
    for k, v in p2_metrics.items():
        print(f"  {k:<22}: {v:.2f}")

    print("\n--- Target Day Weather [T_target] ---")
    for k, v in target_metrics.items():
        print(f"  {k:<26}: {v:.2f}")