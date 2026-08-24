from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class WeatherFetcher:
    """
    Historical meteorological data fetcher via Open-Meteo Archive API.
    Computes interval aggregates, target day surface/upper-air extremes,
    Vapor Pressure Deficit (VPD), Continuous Haines Index (c-Haines),
    and antecedent dry spell duration.
    """

    BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    _ARCHIVE_LATENCY_DAYS: int = 5

    def __init__(
        self,
        timeout: int = 25,
        dry_day_threshold_mm: float = 1.0,
        max_dry_days_lookback: int = 60,
    ):
        self.timeout = timeout
        self.dry_day_threshold_mm = dry_day_threshold_mm
        self.max_dry_days_lookback = max_dry_days_lookback

        # Persistent connection session with automated exponential backoff
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PreemptFire-ML-Pipeline/1.0 (Wildfire Ignition Research)"
        })

        retries = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    
    # Public Interface
    

    def fetch_interval_metrics(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, float]:
        """
        Computes aggregate metrics for an arbitrary multi-day date range (e.g. between satellite passes).
        """
        self._validate_date_availability(end_date)

        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
        ]
        hourly = self._fetch_hourly_data(lat, lon, start_date, end_date, variables)

        temps = np.array(hourly["temperature_2m"], dtype=np.float32)
        rhs = np.array(hourly["relative_humidity_2m"], dtype=np.float32)
        precips = np.array(hourly["precipitation"], dtype=np.float32)
        winds = np.array(hourly["wind_speed_10m"], dtype=np.float32)

        d_start = datetime.strptime(start_date, "%Y-%m-%d")
        d_end = datetime.strptime(end_date, "%Y-%m-%d")
        num_days = max(1, (d_end - d_start).days + 1)

        vpds = self._calc_vpd(temps, rhs)

        return {
            "temperature_mean": float(np.nanmean(temps)),
            "temperature_max": float(np.nanmax(temps)),
            "humidity_mean": float(np.nanmean(rhs)),
            "humidity_min": float(np.nanmin(rhs)),
            "vpd_mean": float(np.nanmean(vpds)),
            "vpd_max": float(np.nanmax(vpds)),
            "precip_daily_rate": float(np.nansum(precips) / num_days),
            "wind_speed_mean": float(np.nanmean(winds)),
        }

    def fetch_target_day_metrics(self, lat: float, lon: float, target_date: str) -> Dict[str, float]:
        """
        Computes comprehensive meteorological features for the target prediction day:
        - Surface extremes & means (T, RH, Wind, Dew Point, Surface Pressure, Precip)
        - Vapor Pressure Deficit (VPD max & mean in kPa)
        - Atmospheric vertical instability: Continuous Haines Index (c-Haines)
        - Antecedent Drought: Days since last significant rain (Precip >= threshold)
        """
        self._validate_date_availability(target_date)

        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "precipitation",
            "temperature_850hPa",
            "temperature_700hPa",
            "dew_point_850hPa",
        ]
        hourly = self._fetch_hourly_data(lat, lon, target_date, target_date, variables)

        temps = np.array(hourly["temperature_2m"], dtype=np.float32)
        rhs = np.array(hourly["relative_humidity_2m"], dtype=np.float32)
        dews = np.array(hourly["dew_point_2m"], dtype=np.float32)
        winds = np.array(hourly["wind_speed_10m"], dtype=np.float32)
        wind_dirs = np.array(hourly["wind_direction_10m"], dtype=np.float32)
        pressures = np.array(hourly["surface_pressure"], dtype=np.float32)
        precips = np.array(hourly["precipitation"], dtype=np.float32)

        # Upper atmosphere variables for vertical instability
        t_850 = np.array(hourly["temperature_850hPa"], dtype=np.float32)
        t_700 = np.array(hourly["temperature_700hPa"], dtype=np.float32)
        dp_850 = np.array(hourly["dew_point_850hPa"], dtype=np.float32)

        # 1. Circular vector mean for meteorological wind direction
        rads = np.radians(wind_dirs)
        mean_sin = np.nanmean(np.sin(rads))
        mean_cos = np.nanmean(np.cos(rads))
        mean_wind_dir = (np.degrees(np.arctan2(mean_sin, mean_cos)) + 360.0) % 360.0

        # 2. Vapor Pressure Deficit (kPa)
        vpds = self._calc_vpd(temps, rhs)

        # 3. Continuous Haines Index (c-Haines) with NaN safety guard
        c_haines_hourly = self._calc_continuous_haines(t_850, t_700, dp_850)
        c_haines_max = float(np.nanmax(c_haines_hourly)) if not np.all(np.isnan(c_haines_hourly)) else 0.0
        c_haines_mean = float(np.nanmean(c_haines_hourly)) if not np.all(np.isnan(c_haines_hourly)) else 0.0

        # 4. Antecedent Dry Spell (Days since rain >= threshold)
        days_since_rain = self._calc_days_since_rain(lat, lon, target_date)

        return {
            "temperature_2m_max": float(np.nanmax(temps)),
            "temperature_2m_min": float(np.nanmin(temps)),
            "relative_humidity_2m_min": float(np.nanmin(rhs)),
            "relative_humidity_2m_mean": float(np.nanmean(rhs)),
            "dew_point_2m_mean": float(np.nanmean(dews)),
            "vpd_kpa_max": float(np.nanmax(vpds)),
            "vpd_kpa_mean": float(np.nanmean(vpds)),
            "c_haines_max": c_haines_max,
            "c_haines_mean": c_haines_mean,
            "wind_speed_10m_max": float(np.nanmax(winds)),
            "wind_direction_10m": float(mean_wind_dir),
            "surface_pressure_mean": float(np.nanmean(pressures)),
            "precipitation_sum": float(np.nansum(precips)),
            "days_since_rain": float(days_since_rain),
        }

    
    # Private Ingestion & Validation
    

    def _validate_date_availability(self, date_str: str) -> None:
        """Raises ValueError if date_str falls within the Open-Meteo archive latency window."""
        d = datetime.strptime(date_str, "%Y-%m-%d")
        latest = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=self._ARCHIVE_LATENCY_DAYS)
        if d > latest:
            raise ValueError(
                f"date {date_str} is within the Open-Meteo archive latency window "
                f"(data available up to {latest.strftime('%Y-%m-%d')}). "
                f"Choose an earlier date."
            )

    def _fetch_hourly_data(self, lat: float, lon: float, start_date: str, end_date: str, variables: list) -> dict:
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "UTC",
        }
        resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            raise ValueError(f"No hourly weather data returned for {lat}, {lon} ({start_date} to {end_date})")

        return data["hourly"]

    
    # Private Thermodynamic & Empirical Formulations
    

    @staticmethod
    def _calc_vpd(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """
        Calculates Vapor Pressure Deficit (VPD in kPa) using the Tetens equation
        with temperature-adaptive coefficients per WMO (2018) guidelines:
          - T >= 0°C : 17.368 / 238.83 (over liquid water)
          - T <  0°C : 17.966 / 247.15 (over ice phase)
        """
        rh_clipped = np.clip(rh, 0.0, 100.0)

        a = np.where(temp_c >= 0.0, 17.368, 17.966).astype(np.float32)
        b = np.where(temp_c >= 0.0, 238.83, 247.15).astype(np.float32)

        e_sat = 0.61078 * np.exp((a * temp_c) / (temp_c + b))
        vpd = e_sat * (1.0 - rh_clipped / 100.0)
        return np.maximum(0.0, vpd)

    @staticmethod
    def _calc_continuous_haines(t_850: np.ndarray, t_700: np.ndarray, dp_850: np.ndarray) -> np.ndarray:
        """
        Computes Continuous Haines Index (c-Haines) following Mills & McCaw (2010).
        c-Haines = CA (Lapse rate / Stability) + CB (Moisture deficit at 850 hPa)
        """
        lapse = t_850 - t_700
        ca = np.clip(0.5 * lapse - 2.0, 0.0, 5.0)

        dp_depress = t_850 - dp_850
        cb = np.clip(0.3333 * dp_depress - 1.0, 0.0, 5.0)

        return ca + cb

    def _calc_days_since_rain(self, lat: float, lon: float, target_date: str) -> int:
        """
        Calculates consecutive dry days prior to target_date.
        Aggregates UTC hourly precipitation into calendar days aligned to
        local solar time via lon-derived integer offset.
        """
        d_target = datetime.strptime(target_date, "%Y-%m-%d")
        d_start = d_target - timedelta(days=self.max_dry_days_lookback)

        hourly = self._fetch_hourly_data(
            lat=lat,
            lon=lon,
            start_date=d_start.strftime("%Y-%m-%d"),
            end_date=(d_target - timedelta(days=1)).strftime("%Y-%m-%d"),
            variables=["precipitation"],
        )

        times_utc = [datetime.fromisoformat(t) for t in hourly["time"]]
        precips = np.array(hourly["precipitation"], dtype=np.float32)

        # Shift timestamps to approximate local solar day
        offset_h = int(round(lon / 15.0))
        times_loc = [t + timedelta(hours=offset_h) for t in times_utc]

        daily_totals: Dict[Any, float] = defaultdict(float)
        for prec, t_loc in zip(precips, times_loc):
            if not np.isnan(prec):
                daily_totals[t_loc.date()] += float(prec)

        dry_days = 0
        check_date = (d_target - timedelta(days=1)).date()
        for _ in range(self.max_dry_days_lookback):
            total = daily_totals.get(check_date, 0.0)
            if total >= self.dry_day_threshold_mm:
                break
            dry_days += 1
            check_date -= timedelta(days=1)

        return dry_days


if __name__ == "__main__":
    lat, lon = 53.5461, -113.4937  # Edmonton

    date_t_prev = "2023-07-01"
    date_t0 = "2023-07-15"

    fetcher = WeatherFetcher(dry_day_threshold_mm=1.5)

    interval_metrics = fetcher.fetch_interval_metrics(lat, lon, start_date=date_t_prev, end_date=date_t0)
    target_metrics = fetcher.fetch_target_day_metrics(lat, lon, target_date=date_t0)

    print(f"\n--- Interval Weather [{date_t_prev} -> {date_t0}] ---")
    for k, v in interval_metrics.items():
        print(f"  {k:<22}: {v:6.2f}")

    print(f"\n--- Target Day Weather [{date_t0}] ---")
    for k, v in target_metrics.items():
        print(f"  {k:<26}: {v:6.2f}")