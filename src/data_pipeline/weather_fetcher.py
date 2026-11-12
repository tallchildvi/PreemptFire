from datetime import datetime, timedelta
import time
from typing import Dict, Optional
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class WeatherFetcher:
    """Fetches single-pass Open-Meteo ERA5 surface meteorology and lookback statistics without indices or Tprev."""

    def __init__(self, rain_threshold_mm: float = 1.5):
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.rain_threshold_mm = rain_threshold_mm
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

    @staticmethod
    def _safe_hourly_array(hourly_dict: dict, var_name: str, n_last: Optional[int] = None) -> np.ndarray:
        """Safely extracts hourly series handling potential nulls and model suffixes."""
        raw_vals = hourly_dict.get(var_name)
        if raw_vals is None:
            raw_vals = hourly_dict.get(f"{var_name}_era5", [])

        arr = np.array([np.nan if v is None else v for p in [raw_vals] for v in p], dtype=np.float32) if raw_vals else np.array([], dtype=np.float32)
        if n_last is not None and len(arr) >= n_last:
            return arr[-n_last:]
        return arr

    def _fetch_raw_archive(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> Dict:
        """Queries Open-Meteo archive in a single HTTP request."""
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "precipitation",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            "daily": ["precipitation_sum"],
            "timezone": "UTC",
        }
        res = self.session.get(self.base_url, params=params, timeout=25)
        res.raise_for_status()
        return res.json()

    @staticmethod
    def _aggregate_slice(
        temp: np.ndarray,
        rh: np.ndarray,
        wind: np.ndarray,
        precip: np.ndarray,
        prefix: str,
    ) -> Dict[str, float]:
        """Aggregates temperature, VPD, humidity, wind, and precipitation over an hourly slice."""
        if len(temp) == 0 or np.isnan(temp).all():
            return {
                f"{prefix}_temp_mean": 0.0,
                f"{prefix}_temp_max": 0.0,
                f"{prefix}_rh_min": 0.0,
                f"{prefix}_vpd_mean": 0.0,
                f"{prefix}_wind_max": 0.0,
                f"{prefix}_precip_sum": 0.0,
            }

        svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
        avp = svp * (rh / 100.0)
        vpd = np.maximum(svp - avp, 0.0)

        return {
            f"{prefix}_temp_mean": float(np.nanmean(temp)),
            f"{prefix}_temp_max": float(np.nanmax(temp)),
            f"{prefix}_rh_min": float(np.nanmin(rh)),
            f"{prefix}_vpd_mean": float(np.nanmean(vpd)),
            f"{prefix}_wind_max": float(np.nanmax(wind)),
            f"{prefix}_precip_sum": float(np.nansum(precip)),
        }

    def fetch_all_weather_metrics(
        self, lat: float, lon: float, target_date: str, t0_date: Optional[str] = None
    ) -> Dict[str, float]:
        """Fetches 60-day weather archive in ONE HTTP call and computes raw meteorological summaries."""
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        history_start_dt = target_dt - timedelta(days=60)
        history_start = history_start_dt.strftime("%Y-%m-%d")

        data = self._fetch_raw_archive(lat, lon, history_start, target_date)

        daily_precip_raw = data.get("daily", {}).get("precipitation_sum", [])
        daily_precip = np.array([0.0 if p is None else p for p in daily_precip_raw], dtype=np.float32)
        target_day_precip = float(daily_precip[-1]) if len(daily_precip) > 0 else 0.0

        days_since_rain = 0.0
        if target_day_precip < self.rain_threshold_mm and len(daily_precip) > 1:
            for p in reversed(daily_precip[:-1]):
                if p >= self.rain_threshold_mm:
                    break
                days_since_rain += 1.0

        hourly = data.get("hourly", {})
        temp_all = self._safe_hourly_array(hourly, "temperature_2m")
        rh_all = self._safe_hourly_array(hourly, "relative_humidity_2m")
        dp_all = self._safe_hourly_array(hourly, "dew_point_2m")
        wind_all = self._safe_hourly_array(hourly, "wind_speed_10m")
        wdir_all = self._safe_hourly_array(hourly, "wind_direction_10m")
        press_all = self._safe_hourly_array(hourly, "surface_pressure")
        precip_all = self._safe_hourly_array(hourly, "precipitation")

        # 1. Target day (last 24 hours)
        h_temp = temp_all[-24:]
        h_rh = rh_all[-24:]
        h_dp = dp_all[-24:]
        h_wind = wind_all[-24:]
        h_wdir = wdir_all[-24:]
        h_press = press_all[-24:]

        svp = 0.61078 * np.exp((17.27 * h_temp) / (h_temp + 237.3))
        avp = svp * (h_rh / 100.0)
        vpd = np.maximum(svp - avp, 0.0)

        valid_wdir = h_wdir[np.isfinite(h_wdir)]
        if len(valid_wdir) > 0:
            sin_mean = float(np.mean(np.sin(np.radians(valid_wdir))))
            cos_mean = float(np.mean(np.cos(np.radians(valid_wdir))))
            wdir_mean = float((np.degrees(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)
        else:
            wdir_mean = 0.0

        target_metrics = {
            "target_temperature_max": float(np.nanmax(h_temp)),
            "target_temperature_min": float(np.nanmin(h_temp)),
            "target_temperature_mean": float(np.nanmean(h_temp)),
            "target_dew_point_mean": float(np.nanmean(h_dp)) if len(h_dp) > 0 else 0.0,
            "target_rh_min": float(np.nanmin(h_rh)),
            "target_rh_mean": float(np.nanmean(h_rh)),
            "target_vpd_max": float(np.nanmax(vpd)),
            "target_vpd_mean": float(np.nanmean(vpd)),
            "target_wind_speed_max": float(np.nanmax(h_wind)),
            "target_wind_speed_mean": float(np.nanmean(h_wind)),
            "target_wind_direction_mean": wdir_mean,
            "target_surface_pressure_mean": float(np.nanmean(h_press)),
            "target_precipitation_sum": target_day_precip,
            "days_since_rain": days_since_rain,
        }

        # 2. Satellite T0 -> Target Date gap interval
        if t0_date:
            t0_dt = datetime.strptime(t0_date, "%Y-%m-%d")
            gap_days = max(1, min(60, (target_dt - t0_dt).days))
        else:
            gap_days = 1
        gap_hours = gap_days * 24

        int1_metrics = self._aggregate_slice(
            temp_all[-gap_hours:],
            rh_all[-gap_hours:],
            wind_all[-gap_hours:],
            precip_all[-gap_hours:],
            prefix="int1_t0_to_target",
        )
        int1_metrics["days_since_t0_pass"] = float(gap_days)

        # 3. 7-day and 30-day synoptic lookback windows
        lb7_metrics = self._aggregate_slice(
            temp_all[-(7 * 24):],
            rh_all[-(7 * 24):],
            wind_all[-(7 * 24):],
            precip_all[-(7 * 24):],
            prefix="lookback_7d",
        )
        lb30_metrics = self._aggregate_slice(
            temp_all[-(30 * 24):],
            rh_all[-(30 * 24):],
            wind_all[-(30 * 24):],
            precip_all[-(30 * 24):],
            prefix="lookback_30d",
        )

        return {
            **target_metrics,
            **int1_metrics,
            **lb7_metrics,
            **lb30_metrics,
        }


if __name__ == "__main__":
    fetcher = WeatherFetcher(rain_threshold_mm=1.5)
    test_lat, test_lon = 41.6986, -7.0011
    test_target_date = "2022-08-08"
    test_t0_date = "2022-08-05"

    print(f"Testing raw WeatherFetcher for ({test_lat}, {test_lon}) on {test_target_date}...")
    t_start = time.perf_counter()
    metrics = fetcher.fetch_all_weather_metrics(
        lat=test_lat, lon=test_lon, target_date=test_target_date, t0_date=test_t0_date
    )
    elapsed = time.perf_counter() - t_start

    print(f"Completed in {elapsed:.2f}s (Total 1D weather features: {len(metrics)})\n")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<32}: {v:>10.4f}")