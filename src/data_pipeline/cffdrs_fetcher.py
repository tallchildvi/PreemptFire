from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Tuple
from numba import njit
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


@njit(cache=True)
def _njit_get_day_length_factors(
    lat: float,
    month: int,
    dmc_north: np.ndarray,
    dmc_south: np.ndarray,
    dc_north: np.ndarray,
    dc_south: np.ndarray,
    dc_lf_eq: float,
) -> Tuple[float, float]:
    m_idx = month - 1
    if m_idx < 0:
        m_idx = 0
    elif m_idx > 11:
        m_idx = 11

    if lat > 15.0:
        return dmc_north[m_idx], dc_north[m_idx]
    elif lat < -15.0:
        return dmc_south[m_idx], dc_south[m_idx]
    else:
        return 9.0, dc_lf_eq


@njit(cache=True)
def _njit_calc_ffmc(
    temp: float, rh: float, wind_kmh: float, rain_24h: float, prev_ffmc: float
) -> float:
    m_0 = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

    if rain_24h > 0.5:
        r_a = rain_24h - 0.5
        if m_0 > 150.0:
            m_r = (
                m_0
                + 42.5 * r_a * math.exp(-100.0 / (251.0 - m_0)) * (1.0 - math.exp(-6.93 / r_a))
                + 0.0015 * ((m_0 - 150.0) ** 2) * math.sqrt(r_a)
            )
        else:
            m_r = m_0 + 42.5 * r_a * math.exp(-100.0 / (251.0 - m_0)) * (1.0 - math.exp(-6.93 / r_a))
        m_0 = min(250.0, m_r)

    e_d = (
        0.942 * (rh**0.679)
        + 11.0 * math.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    )
    e_w = (
        0.618 * (rh**0.753)
        + 10.0 * math.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    )

    if m_0 > e_d:
        k_a = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind_kmh) * (1.0 - (rh / 100.0) ** 8)
        k_d = k_a * 0.581 * math.exp(0.0365 * temp)
        m = e_d + (m_0 - e_d) * (10.0 ** (-k_d))
    elif m_0 < e_w:
        k_b = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind_kmh) * (
            1.0 - ((100.0 - rh) / 100.0) ** 8
        )
        k_w = k_b * 0.581 * math.exp(0.0365 * temp)
        m = e_w - (e_w - m_0) * (10.0 ** (-k_w))
    else:
        m = m_0

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    if ffmc < 0.0:
        return 0.0
    elif ffmc > 101.0:
        return 101.0
    return ffmc


@njit(cache=True)
def _njit_calc_dmc(
    temp: float, rh: float, rain_24h: float, l_e: float, prev_dmc: float
) -> float:
    if rain_24h > 1.5:
        r_e = 0.92 * rain_24h - 1.27
        m_0 = 20.0 + math.exp(5.6348 - prev_dmc / 43.43)
        if prev_dmc <= 33.0:
            b = 100.0 / (0.5 + 0.3 * prev_dmc)
        elif prev_dmc <= 65.0:
            b = 14.0 - 1.3 * math.log(prev_dmc)
        else:
            b = 6.2 * math.log(prev_dmc) - 17.2

        m_r = m_0 + 1000.0 * r_e / (48.77 + b * r_e)
        prev_dmc = max(0.0, 244.72 - 43.43 * math.log(max(1.0, m_r - 20.0)))

    t_k = max(-1.1, temp)
    k = 1.894 * (t_k + 1.1) * (100.0 - rh) * l_e * 1e-4
    return max(0.0, prev_dmc + max(0.0, k))


@njit(cache=True)
def _njit_calc_dc(
    temp: float, rain_24h: float, l_f: float, prev_dc: float
) -> float:
    if rain_24h > 2.8:
        r_d = 0.83 * rain_24h - 1.27
        q_0 = 800.0 * math.exp(-prev_dc / 400.0)
        q_r = q_0 + 3.937 * r_d
        prev_dc = max(0.0, 400.0 * math.log(800.0 / max(1.0, q_r)))

    t_k = max(-2.8, temp)
    v = 0.36 * (t_k + 2.8) + l_f
    return max(0.0, prev_dc + 0.5 * max(0.0, v))


@njit(cache=True)
def _njit_calc_isi(wind_kmh: float, ffmc: float) -> float:
    f_w = math.exp(0.05039 * wind_kmh)
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    f_f = 91.9 * math.exp(-0.1386 * m) * (1.0 + (m**5.31) / 4.93e7)
    return 0.208 * f_w * f_f


@njit(cache=True)
def _njit_calc_bui(dmc: float, dc: float) -> float:
    if dmc <= 0.0 and dc <= 0.0:
        return 0.0

    denom = dmc + 0.4 * dc
    if denom <= 1e-6:
        return 0.0

    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / denom
    else:
        bui = dmc - (1.0 - 0.8 * dc / denom) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(0.0, bui)


@njit(cache=True)
def _njit_calc_fwi(isi: float, bui: float) -> float:
    if bui <= 80.0:
        f_d = 0.626 * (bui**0.80) + 0.1
    else:
        f_d = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))

    b = 0.1 * isi * f_d
    if b > 1.0:
        ln_s = 2.72 * ((0.434 * math.log(b)) ** 0.647)
        return math.exp(ln_s)
    return b


@njit(cache=True)
def _cffdrs_spinup_kernel(
    months: np.ndarray,
    temps: np.ndarray,
    rhs: np.ndarray,
    winds: np.ndarray,
    rains: np.ndarray,
    lat: float,
    init_ffmc: float,
    init_dmc: float,
    init_dc: float,
    dmc_north: np.ndarray,
    dmc_south: np.ndarray,
    dc_north: np.ndarray,
    dc_south: np.ndarray,
    dc_lf_eq: float,
) -> Tuple[float, float, float, float, float, float]:
    ffmc = init_ffmc
    dmc = init_dmc
    dc = init_dc

    n = len(months)
    for i in range(n):
        month = months[i]
        t = temps[i]
        rh = rhs[i]
        w = winds[i]
        r = rains[i]

        l_e, l_f = _njit_get_day_length_factors(
            lat, month, dmc_north, dmc_south, dc_north, dc_south, dc_lf_eq
        )

        ffmc = _njit_calc_ffmc(t, rh, w, r, ffmc)
        dmc = _njit_calc_dmc(t, rh, r, l_e, dmc)
        dc = _njit_calc_dc(t, r, l_f, dc)

    target_wind = winds[n - 1] if n > 0 else 0.0
    isi = _njit_calc_isi(target_wind, ffmc)
    bui = _njit_calc_bui(dmc, dc)
    fwi = _njit_calc_fwi(isi, bui)

    return ffmc, dmc, dc, isi, bui, fwi



# Public Orchestrator Class



class CFFDRSFetcher:
    """
    Global autonomous fetcher and computer for the Canadian Forest Fire Danger Rating System (CFFDRS)[cite: 2].
    Implements standard global day-length and evaporative adjustments (Northern, Equatorial, Southern)[cite: 2]
    accelerated with cached Numba JIT.
    """

    BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    # Static daylight factor arrays for JIT consumption[cite: 2]
    DMC_DAY_LENGTH_NORTH: np.ndarray = np.array(
        [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],
        dtype=np.float64,
    )
    DMC_DAY_LENGTH_SOUTH: np.ndarray = np.array(
        [12.4, 10.9, 9.4, 8.0, 7.0, 6.0, 6.5, 7.5, 9.0, 12.8, 13.9, 13.9],
        dtype=np.float64,
    )

    DC_DAY_LENGTH_NORTH: np.ndarray = np.array(
        [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],
        dtype=np.float64,
    )
    DC_DAY_LENGTH_SOUTH: np.ndarray = np.array(
        [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8],
        dtype=np.float64,
    )

    _DC_LF_EQUATORIAL: float = float(
        np.mean(DC_DAY_LENGTH_NORTH) * 0.5 + np.mean(DC_DAY_LENGTH_SOUTH) * 0.5
    )
    _ARCHIVE_LATENCY_DAYS: int = 5

    def __init__(
        self,
        timeout: int = 20,
        spinup_days: int = 90,
        default_ffmc: float = 85.0,
        default_dmc: float = 6.0,
        default_dc: float = 15.0,
    ):
        self.timeout = timeout
        self.spinup_days = spinup_days
        self.default_ffmc = float(default_ffmc)
        self.default_dmc = float(default_dmc)
        self.default_dc = float(default_dc)

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "PreemptFireWildfirePipeline/1.0 (Research Project)"}
        )
        retries = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retries, pool_connections=10, pool_maxsize=10
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_cffdrs_metrics(
        self, lat: float, lon: float, date_t0: str
    ) -> Dict[str, float]:
        """
        Fetches 90-day history before date_t0, runs the Numba cached JIT spin-up,
        and returns all 6 calibrated CFFDRS indices[cite: 2].
        """
        d_end = datetime.strptime(date_t0, "%Y-%m-%d")
        latest_available = datetime.now(timezone.utc).replace(
            tzinfo=None
        ) - timedelta(days=self._ARCHIVE_LATENCY_DAYS)

        if d_end > latest_available:
            raise ValueError(
                f"date_t0 {date_t0} is within Open-Meteo archive latency window "
                f"(data available up to {latest_available.strftime('%Y-%m-%d')}). "
                f"Choose an earlier date."
            )

        d_start = d_end - timedelta(days=self.spinup_days + 2)
        start_str = d_start.strftime("%Y-%m-%d")
        end_str = d_end.strftime("%Y-%m-%d")

        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
        ]

        hourly_data = self._fetch_hourly_data(
            lat, lon, start_str, end_str, variables
        )
        months, temps, rhs, winds, rains = self._extract_noon_arrays(
            hourly_data, lon
        )

        if len(months) == 0:
            raise ValueError(
                f"No valid noon records extracted for {lat}, {lon} ending at {date_t0}"
            )

        ffmc, dmc, dc, isi, bui, fwi = _cffdrs_spinup_kernel(
            months=months,
            temps=temps,
            rhs=rhs,
            winds=winds,
            rains=rains,
            lat=float(lat),
            init_ffmc=self.default_ffmc,
            init_dmc=self.default_dmc,
            init_dc=self.default_dc,
            dmc_north=self.DMC_DAY_LENGTH_NORTH,
            dmc_south=self.DMC_DAY_LENGTH_SOUTH,
            dc_north=self.DC_DAY_LENGTH_NORTH,
            dc_south=self.DC_DAY_LENGTH_SOUTH,
            dc_lf_eq=self._DC_LF_EQUATORIAL,
        )

        return {
            "cffdrs_ffmc": float(ffmc),
            "cffdrs_dmc": float(dmc),
            "cffdrs_dc": float(dc),
            "cffdrs_isi": float(isi),
            "cffdrs_bui": float(bui),
            "cffdrs_fwi": float(fwi),
        }

    
    # Private Ingestion & Solar Time Extraction
    

    def _fetch_hourly_data(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: list,
    ) -> dict:
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "UTC",
        }
        resp = self.session.get(
            self.BASE_URL, params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            raise ValueError(
                f"No hourly weather data returned for {lat}, {lon} ({start_date} to {end_date})"
            )

        return data["hourly"]

    @staticmethod
    def _solar_noon_utc_hour(date: datetime, lon: float) -> float:
        doy = date.timetuple().tm_yday
        b = 2.0 * np.pi * (doy - 1) / 365.0

        eot_minutes = 229.18 * (
            0.000075
            + 0.001868 * np.cos(b)
            - 0.032077 * np.sin(b)
            - 0.014615 * np.cos(2.0 * b)
            - 0.04089 * np.sin(2.0 * b)
        )
        solar_noon_utc = 12.0 - (eot_minutes / 60.0) - (lon / 15.0)
        return float(solar_noon_utc % 24.0)

    def _extract_noon_arrays(
        self, hourly: dict, lon: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        times_utc = [datetime.fromisoformat(t) for t in hourly["time"]]

        temps = np.array(hourly["temperature_2m"], dtype=np.float64)
        rhs = np.array(hourly["relative_humidity_2m"], dtype=np.float64)
        winds = np.array(hourly["wind_speed_10m"], dtype=np.float64)
        precips = np.array(hourly["precipitation"], dtype=np.float64)

        date_to_indices: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(times_utc):
            date_to_indices[t.date()].append((i, t.hour))

        months_list: List[int] = []
        temps_list: List[float] = []
        rhs_list: List[float] = []
        winds_list: List[float] = []
        rains_list: List[float] = []

        for date in sorted(date_to_indices.keys()):
            solar_noon_utc = self._solar_noon_utc_hour(
                datetime(date.year, date.month, date.day), lon
            )
            entries = date_to_indices[date]
            best_i, _ = min(
                entries,
                key=lambda e: abs((e[1] - solar_noon_utc + 12) % 24 - 12),
            )

            t_noon = temps[best_i]
            if t_noon < -1.1 or best_i < 23:
                continue

            rain_24h = float(np.nansum(precips[best_i - 23 : best_i + 1]))

            months_list.append(date.month)
            temps_list.append(t_noon)
            rhs_list.append(float(np.clip(rhs[best_i], 1.0, 100.0)))
            winds_list.append(max(0.0, winds[best_i]))
            rains_list.append(max(0.0, rain_24h))

        return (
            np.array(months_list, dtype=np.int32),
            np.array(temps_list, dtype=np.float64),
            np.array(rhs_list, dtype=np.float64),
            np.array(winds_list, dtype=np.float64),
            np.array(rains_list, dtype=np.float64),
        )


if __name__ == "__main__":
    fetcher = CFFDRSFetcher(spinup_days=90)

    GLOBAL_TEST_SITES = {
        "Canada (Alberta Boreal)": (56.7264, -111.3803, "2023-07-15"),
        "Australia (NSW Bushfire season)": (-33.8688, 151.2093, "2020-01-05"),
        "Greece (Mediterranean summer)": (38.0408, 23.8202, "2023-08-22"),
        "Indonesia (Equatorial peatland)": (-0.7893, 113.9213, "2023-09-15"),
    }

    for name, (lat, lon, target_date) in GLOBAL_TEST_SITES.items():
        print(f"\nSite: {name} [{lat}, {lon}] at {target_date}")
        metrics = fetcher.fetch_cffdrs_metrics(lat, lon, date_t0=target_date)
        for k, v in metrics.items():
            print(f"  {k:<16}: {v:6.2f}")