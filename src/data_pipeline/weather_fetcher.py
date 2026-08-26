from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import requests


class WeatherFetcher:
    """fetches multi-interval open-meteo metrics and upper-air atmospheric stability."""

    def __init__(self, rain_threshold_mm: float = 1.5):
        self.rain_threshold_mm = rain_threshold_mm
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

    def compute_continuous_haines(
        self, t850: np.ndarray, t700: np.ndarray, dp850: np.ndarray
    ) -> np.ndarray:
        """calculates continuous haines index (mills 2005 / 2008)."""
        ca = np.clip(0.5 * (t850 - t700) - 2.0, 0.0, None)
        cb = np.clip(0.3333 * (t850 - dp850) - 1.0, 0.0, 9.0)
        return ca + cb

    def _fetch_raw_archive(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> Dict:
        """helper to query open-meteo archive with all surface and pressure level bands."""
        params = {
            "latitude": lat,
            "longitude": lon,
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
                "temperature_850hPa",
                "temperature_700hPa",
                "dew_point_850hPa",
            ],
            "daily": ["precipitation_sum"],
            "timezone": "UTC",
        }
        res = requests.get(self.base_url, params=params, timeout=20)
        res.raise_for_status()
        return res.json()

    def _aggregate_hourly_slice(self, raw_data: Dict, prefix: str) -> Dict[str, float]:
        """aggregates temperature, vpd, humidity, wind and rain over an arbitrary hourly slice."""
        hourly = raw_data.get("hourly", {})
        temp = np.array(hourly.get("temperature_2m", []), dtype=np.float32)
        rh = np.array(hourly.get("relative_humidity_2m", []), dtype=np.float32)
        wind = np.array(hourly.get("wind_speed_10m", []), dtype=np.float32)
        precip = np.array(hourly.get("precipitation", []), dtype=np.float32)

        if len(temp) == 0:
            return {
                f"{prefix}_temp_mean": 0.0,
                f"{prefix}_temp_max": 0.0,
                f"{prefix}_rh_min": 0.0,
                f"{prefix}_vpd_mean": 0.0,
                f"{prefix}_wind_max": 0.0,
                f"{prefix}_precip_sum": 0.0,
            }

        # calculate vapor pressure deficit (kpa)
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

    def fetch_target_day_metrics(
        self, lat: float, lon: float, target_date: str
    ) -> Dict[str, float]:
        """fetches 24h conditions on target date and calculates days since rain lookback."""
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        history_start = (target_dt - timedelta(days=60)).strftime("%Y-%m-%d")

        data = self._fetch_raw_archive(lat, lon, history_start, target_date)
        daily_precip = np.array(data["daily"]["precipitation_sum"], dtype=np.float32)
        target_day_precip = float(daily_precip[-1])

        # calculate drought days
        days_since_rain = 0.0
        if target_day_precip < self.rain_threshold_mm:
            for p in reversed(daily_precip[:-1]):
                if p >= self.rain_threshold_mm:
                    break
                days_since_rain += 1.0

        # target day slice (last 24 hours)
        t850 = np.array(data["hourly"]["temperature_850hPa"][-24:], dtype=np.float32)
        t700 = np.array(data["hourly"]["temperature_700hPa"][-24:], dtype=np.float32)
        dp850 = np.array(data["hourly"]["dew_point_850hPa"][-24:], dtype=np.float32)

        c_haines_mean, c_haines_max = 0.0, 0.0
        if not np.isnan(t850).all() and not np.isnan(t700).all():
            ch_arr = self.compute_continuous_haines(t850, t700, dp850)
            c_haines_mean = float(np.nanmean(ch_arr))
            c_haines_max = float(np.nanmax(ch_arr))

        h_temp = np.array(data["hourly"]["temperature_2m"][-24:], dtype=np.float32)
        h_rh = np.array(data["hourly"]["relative_humidity_2m"][-24:], dtype=np.float32)
        h_wind = np.array(data["hourly"]["wind_speed_10m"][-24:], dtype=np.float32)
        h_wdir = np.array(data["hourly"]["wind_direction_10m"][-24:], dtype=np.float32)
        h_press = np.array(data["hourly"]["surface_pressure"][-24:], dtype=np.float32)

        svp = 0.61078 * np.exp((17.27 * h_temp) / (h_temp + 237.3))
        avp = svp * (h_rh / 100.0)
        vpd = np.maximum(svp - avp, 0.0)

        return {
            "target_temperature_max": float(np.nanmax(h_temp)),
            "target_temperature_min": float(np.nanmin(h_temp)),
            "target_rh_min": float(np.nanmin(h_rh)),
            "target_rh_mean": float(np.nanmean(h_rh)),
            "target_vpd_max": float(np.nanmax(vpd)),
            "target_vpd_mean": float(np.nanmean(vpd)),
            "target_c_haines_max": c_haines_max,
            "target_c_haines_mean": c_haines_mean,
            "target_wind_speed_max": float(np.nanmax(h_wind)),
            "target_wind_direction_mean": float(np.nanmean(h_wdir)),
            "target_surface_pressure_mean": float(np.nanmean(h_press)),
            "target_precipitation_sum": target_day_precip,
            "days_since_rain": days_since_rain,
        }

    def fetch_tri_interval_metrics(
        self, lat: float, lon: float, target_date: str, t0_date: str, tprev_date: str
    ) -> Dict[str, float]:
        """fetches and aggregates weather across 3 sequential non-overlapping intervals."""
        t_tgt_dt = datetime.strptime(target_date, "%Y-%m-%d")
        t0_dt = datetime.strptime(t0_date, "%Y-%m-%d")
        tprev_dt = datetime.strptime(tprev_date, "%Y-%m-%d")

        delta_days = max((t0_dt - tprev_dt).days, 1)
        t_base_dt = tprev_dt - timedelta(days=delta_days)

        start_str = t_base_dt.strftime("%Y-%m-%d")
        end_str = t_tgt_dt.strftime("%Y-%m-%d")

        raw_all = self._fetch_raw_archive(lat, lon, start_str, end_str)

        # interval 1: [t0 -> target_date]
        d1 = self._fetch_raw_archive(lat, lon, t0_date, target_date)
        m_int1 = self._aggregate_hourly_slice(d1, prefix="int1_t0_to_target")

        # interval 2: [tprev -> t0]
        d2 = self._fetch_raw_archive(lat, lon, tprev_date, t0_date)
        m_int2 = self._aggregate_hourly_slice(d2, prefix="int2_tprev_to_t0")

        # interval 3: [t_base -> tprev]
        d3 = self._fetch_raw_archive(lat, lon, start_str, tprev_date)
        m_int3 = self._aggregate_hourly_slice(d3, prefix="int3_baseline")

        return {
            **m_int1,
            **m_int2,
            **m_int3,
            "interval_delta_days": float(delta_days),
        }

