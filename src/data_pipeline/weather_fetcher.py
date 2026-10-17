from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import requests


class WeatherFetcher:
    """fetches multi-interval open-meteo metrics and upper-air atmospheric stability."""

    def __init__(self, rain_threshold_mm: float = 1.5):
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.rain_threshold_mm = rain_threshold_mm
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

    def compute_continuous_haines(
        self, t850: np.ndarray, t700: np.ndarray, dp850: np.ndarray
    ) -> np.ndarray:
        """calculates continuous haines index (mills 2005 / 2008)."""
        ca = np.clip(0.5 * (t850 - t700) - 2.0, 0.0, None)
        cb = np.clip(0.3333 * (t850 - dp850) - 1.0, 0.0, 9.0)
        return ca + cb

    @staticmethod
    def _compute_dew_point(temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """calculates dew point from temperature (C) and relative humidity (%) via magnus formula."""
        rh_clipped = np.clip(rh, 1.0, 100.0)
        a, b = 17.27, 237.3
        alpha = ((a * temp) / (b + temp)) + np.log(rh_clipped / 100.0)
        return (b * alpha) / (a - alpha)

    @staticmethod
    def _safe_hourly_array(hourly_dict: dict, var_name: str, n_last: Optional[int] = None) -> np.ndarray:
        """safely extracts hourly series handling potential nulls and model suffixes."""
        raw_vals = hourly_dict.get(var_name)
        if raw_vals is None:
            raw_vals = hourly_dict.get(f"{var_name}_era5", [])
        
        arr = np.array([np.nan if v is None else v for v in raw_vals], dtype=np.float32)
        if n_last is not None and len(arr) >= n_last:
            return arr[-n_last:]
        return arr

    def _fetch_raw_archive(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> Dict:
        """helper to query open-meteo archive with all surface and pressure level bands via era5."""
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
            ],
            "daily": ["precipitation_sum"],
            "timezone": "UTC",
        }
        res = self.session.get(self.base_url, params=params, timeout=20)
        res.raise_for_status()
        return res.json()

    def _aggregate_hourly_slice(self, raw_data: Dict, prefix: str) -> Dict[str, float]:
        """aggregates temperature, vpd, humidity, wind and rain over an arbitrary hourly slice."""
        hourly = raw_data.get("hourly", {})
        temp = self._safe_hourly_array(hourly, "temperature_2m")
        rh = self._safe_hourly_array(hourly, "relative_humidity_2m")
        wind = self._safe_hourly_array(hourly, "wind_speed_10m")
        precip = self._safe_hourly_array(hourly, "precipitation")

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
        daily_precip_raw = data.get("daily", {}).get("precipitation_sum", [])
        daily_precip = np.array([0.0 if p is None else p for p in daily_precip_raw], dtype=np.float32)
        target_day_precip = float(daily_precip[-1]) if len(daily_precip) > 0 else 0.0

        # calculate drought days
        days_since_rain = 0.0
        if target_day_precip < self.rain_threshold_mm and len(daily_precip) > 1:
            for p in reversed(daily_precip[:-1]):
                if p >= self.rain_threshold_mm:
                    break
                days_since_rain += 1.0

        hourly = data.get("hourly", {})

        # upper-air pressure levels for continuous haines (last 24 hours)
        t850 = self._safe_hourly_array(hourly, "temperature_850hPa", n_last=24)
        t700 = self._safe_hourly_array(hourly, "temperature_700hPa", n_last=24)
        rh850 = self._safe_hourly_array(hourly, "relative_humidity_850hPa", n_last=24)

        c_haines_mean, c_haines_max = 0.0, 0.0
        if len(t850) > 0 and len(t700) > 0 and len(rh850) > 0:
            if not np.isnan(t850).all() and not np.isnan(t700).all() and not np.isnan(rh850).all():
                dp850 = self._compute_dew_point(t850, rh850)
                ch_arr = self.compute_continuous_haines(t850, t700, dp850)
                if not np.isnan(ch_arr).all():
                    c_haines_mean = float(np.nanmean(ch_arr))
                    c_haines_max = float(np.nanmax(ch_arr))

        # surface conditions (last 24 hours)
        h_temp = self._safe_hourly_array(hourly, "temperature_2m", n_last=24)
        h_rh = self._safe_hourly_array(hourly, "relative_humidity_2m", n_last=24)
        h_wind = self._safe_hourly_array(hourly, "wind_speed_10m", n_last=24)
        h_wdir = self._safe_hourly_array(hourly, "wind_direction_10m", n_last=24)
        h_press = self._safe_hourly_array(hourly, "surface_pressure", n_last=24)

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

if __name__ == "__main__":
    import time

    fetcher = WeatherFetcher(rain_threshold_mm=1.5)

    # Тестова точка: пожежа на кордоні Іспанії та Португалії (серпень 2022)
    test_lat = 41.6986
    test_lon = -7.0011
    test_target_date = "2022-08-09"
    test_t0_date = "2022-08-07"
    test_tprev_date = "2022-07-13"

    print(f"=== Testing WeatherFetcher for [{test_lat}, {test_lon}] on {test_target_date} ===")
    t_start = time.perf_counter()

    # 1. Запит добових метрик дня пожежі (приземні + ізобаричні рівні 850/700 гПа)
    print("\n1. Fetching target day metrics (surface + upper-air levels)...")
    target_metrics = fetcher.fetch_target_day_metrics(
        lat=test_lat, lon=test_lon, target_date=test_target_date
    )

    # 2. Запит триінтервальної динаміки посухи та погоди
    print("2. Fetching tri-interval metrics (baseline -> tprev -> t0 -> target)...")
    tri_metrics = fetcher.fetch_tri_interval_metrics(
        lat=test_lat,
        lon=test_lon,
        target_date=test_target_date,
        t0_date=test_t0_date,
        tprev_date=test_tprev_date,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nCompleted in {elapsed:.2f}s")

    # Вивід метрик цільового дня
    print("\n")
    print(" TARGET DAY METRICS (Last 24h)")

    for k, v in sorted(target_metrics.items()):
        print(f"  {k:<32}: {v:>12.4f}")

    # Окрема перевірка індексу нестабільності Хейнса
    ch_max = target_metrics.get("target_c_haines_max", 0.0)
    ch_mean = target_metrics.get("target_c_haines_mean", 0.0)
    print("\n")
    if ch_max > 0.0:
        print(f"  [VALIDATION: OK] Continuous Haines Index: mean={ch_mean:.2f}, max={ch_max:.2f}")
    else:
        print("  [VALIDATION: FAILED] Continuous Haines is 0.0000 (check ERA5 upper-air response)")

    print(" TRI-INTERVAL DYNAMICS")
    print("\n")    
    for k, v in sorted(tri_metrics.items()):
        print(f"  {k:<32}: {v:>12.4f}")