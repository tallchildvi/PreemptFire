from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
import numpy as np
import requests


class CFFDRSFetcher:
    """
    Global autonomous fetcher and computer for the Canadian Forest Fire Danger Rating System (CFFDRS).
    Implements standard global day-length and evaporative adjustments (Northern, Equatorial, Southern)
    following Van Wagner (1987).
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Day-length adjustment factors for DMC (Le) by month (Jan -> Dec)
    DMC_DAY_LENGTH_NORTH: List[float] = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    DMC_DAY_LENGTH_SOUTH: List[float] = [12.4, 10.9, 9.4, 8.0, 7.0, 6.0, 6.5, 7.5, 9.0, 12.8, 13.9, 13.9]

    # Seasonal evaporative factors for DC (Lf) by month (Jan -> Dec)
    DC_DAY_LENGTH_NORTH: List[float] = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    DC_DAY_LENGTH_SOUTH: List[float] = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]

    # Equatorial Lf: mean of north + south annual tables (≈ 1.39)
    _DC_LF_EQUATORIAL: float = float(
        np.mean(DC_DAY_LENGTH_NORTH) * 0.5 + np.mean(DC_DAY_LENGTH_SOUTH) * 0.5
    )

    # Open-Meteo archive latency buffer (days)
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
        self.default_ffmc = default_ffmc
        self.default_dmc = default_dmc
        self.default_dc = default_dc

    def fetch_cffdrs_metrics(self, lat: float, lon: float, date_t0: str) -> Dict[str, float]:
        """
        Public single-call interface.
        Fetches 90-day history before date_t0, runs the sequential Van Wagner spin-up,
        and returns all 6 CFFDRS indices calibrated for date_t0 globally.
        """
        d_end = datetime.strptime(date_t0, "%Y-%m-%d")

        latest_available = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=self._ARCHIVE_LATENCY_DAYS)
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

        hourly_data = self._fetch_hourly_data(lat, lon, start_str, end_str, variables)
        noon_records = self._extract_noon_records(hourly_data, lon)

        if not noon_records:
            raise ValueError(f"No valid noon records extracted for {lat}, {lon} ending at {date_t0}")

        state = {
            "FFMC": self.default_ffmc,
            "DMC": self.default_dmc,
            "DC": self.default_dc,
            "ISI": 0.0,
            "BUI": 0.0,
            "FWI": 0.0,
        }

        for record in noon_records:
            state = self._step(state, record, lat)

        return {
            "cffdrs_ffmc": float(state["FFMC"]),
            "cffdrs_dmc": float(state["DMC"]),
            "cffdrs_dc": float(state["DC"]),
            "cffdrs_isi": float(state["ISI"]),
            "cffdrs_bui": float(state["BUI"]),
            "cffdrs_fwi": float(state["FWI"]),
        }


    # Private Ingestion & Local Time Alignment


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

    @staticmethod
    def _solar_noon_utc_hour(date: datetime, lon: float) -> float:
        """
        Estimates solar noon in UTC using Spencer's (1971) Equation of Time.
        Error < 2 minutes — sufficient for selecting the correct hourly slot.
        """
        doy = date.timetuple().tm_yday
        b = 2.0 * np.pi * (doy - 1) / 365.0

        eot_minutes = 229.18 * (
            0.000075
            + 0.001868 * np.cos(b)
            - 0.032077 * np.sin(b)
            - 0.014615 * np.cos(2.0 * b)
            - 0.04089  * np.sin(2.0 * b)
        )
        solar_noon_utc = 12.0 - (eot_minutes / 60.0) - (lon / 15.0)
        return float(solar_noon_utc % 24.0)

    def _extract_noon_records(self, hourly: dict, lon: float) -> List[Dict[str, Any]]:
        times_utc = [datetime.fromisoformat(t) for t in hourly["time"]]

        temps   = np.array(hourly["temperature_2m"],       dtype=np.float32)
        rhs     = np.array(hourly["relative_humidity_2m"], dtype=np.float32)
        winds   = np.array(hourly["wind_speed_10m"],       dtype=np.float32)
        precips = np.array(hourly["precipitation"],        dtype=np.float32)

        date_to_indices: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(times_utc):
            date_to_indices[t.date()].append((i, t.hour))

        noon_records: List[Dict[str, Any]] = []
        sorted_dates = sorted(date_to_indices.keys())

        for date in sorted_dates:
            solar_noon_utc = self._solar_noon_utc_hour(
                datetime(date.year, date.month, date.day), lon
            )
            entries = date_to_indices[date]

            # Pick the hourly slot closest to true solar noon
            best_i, _ = min(entries, key=lambda e: abs((e[1] - solar_noon_utc + 12) % 24 - 12))

            # Freeze / snow-cover guard — skip day, hold codes unchanged
            t_noon = float(temps[best_i])
            if t_noon < -1.1:
                continue

            # Need at least 23 prior slots in the UTC array to form a full 24h sum
            if best_i < 23:
                continue

            # Strict 24h antecedent rainfall ending at the solar-noon UTC slot.
            # The UTC array is uniform hourly → [best_i - 23 : best_i + 1] is
            # always exactly 24 hours regardless of longitude or timezone offset.
            rain_24h = float(np.nansum(precips[best_i - 23 : best_i + 1]))

            noon_records.append({
                "date":          date,
                "month":         date.month,
                "temp_noon":     t_noon,
                "rh_noon":       float(np.clip(rhs[best_i], 1.0, 100.0)),
                "wind_noon_kmh": max(0.0, float(winds[best_i])),
                "rain_24h_mm":   max(0.0, rain_24h),
            })

        return noon_records


    # Private Physical Formulations (Van Wagner, 1987)


    def _get_day_length_factors(self, lat: float, month: int) -> Tuple[float, float]:
        """Returns standard global (Le, Lf) daylight factors based on latitude zones."""
        m_idx = int(np.clip(month, 1, 12)) - 1
        if lat > 15.0:
            return self.DMC_DAY_LENGTH_NORTH[m_idx], self.DC_DAY_LENGTH_NORTH[m_idx]
        elif lat < -15.0:
            return self.DMC_DAY_LENGTH_SOUTH[m_idx], self.DC_DAY_LENGTH_SOUTH[m_idx]
        else:
            # Equatorial / Tropical zone (-15° <= lat <= 15°)
            return 9.0, self._DC_LF_EQUATORIAL

    def _calc_ffmc(self, temp: float, rh: float, wind_kmh: float, rain_24h: float, prev_ffmc: float) -> float:
        m_0 = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

        if rain_24h > 0.5:
            r_a = rain_24h - 0.5
            if m_0 > 150.0:
                m_r = (
                    m_0
                    + 42.5 * r_a * np.exp(-100.0 / (251.0 - m_0)) * (1.0 - np.exp(-6.93 / r_a))
                    + 0.0015 * ((m_0 - 150.0) ** 2) * np.sqrt(r_a)
                )
            else:
                m_r = m_0 + 42.5 * r_a * np.exp(-100.0 / (251.0 - m_0)) * (1.0 - np.exp(-6.93 / r_a))
            m_0 = min(250.0, m_r)

        e_d = (
            0.942 * (rh ** 0.679)
            + 11.0 * np.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        )
        e_w = (
            0.618 * (rh ** 0.753)
            + 10.0 * np.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        )

        if m_0 > e_d:
            k_a = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - (rh / 100.0) ** 8)
            k_d = k_a * 0.581 * np.exp(0.0365 * temp)
            m = e_d + (m_0 - e_d) * (10.0 ** (-k_d))
        elif m_0 < e_w:
            k_b = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (
                1.0 - ((100.0 - rh) / 100.0) ** 8
            )
            k_w = k_b * 0.581 * np.exp(0.0365 * temp)
            m = e_w - (e_w - m_0) * (10.0 ** (-k_w))
        else:
            m = m_0

        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        return float(np.clip(ffmc, 0.0, 101.0))

    def _calc_dmc(self, temp: float, rh: float, rain_24h: float, l_e: float, prev_dmc: float) -> float:
        if rain_24h > 1.5:
            r_e = 0.92 * rain_24h - 1.27
            m_0 = 20.0 + np.exp(5.6348 - prev_dmc / 43.43)
            if prev_dmc <= 33.0:
                b = 100.0 / (0.5 + 0.3 * prev_dmc)
            elif prev_dmc <= 65.0:
                b = 14.0 - 1.3 * np.log(prev_dmc)
            else:
                b = 6.2 * np.log(prev_dmc) - 17.2

            m_r = m_0 + 1000.0 * r_e / (48.77 + b * r_e)
            prev_dmc = max(0.0, 244.72 - 43.43 * np.log(max(1.0, m_r - 20.0)))

        t_k = max(-1.1, temp)
        k = 1.894 * (t_k + 1.1) * (100.0 - rh) * l_e * 1e-4
        return float(max(0.0, prev_dmc + max(0.0, k)))

    def _calc_dc(self, temp: float, rain_24h: float, l_f: float, prev_dc: float) -> float:
        if rain_24h > 2.8:
            r_d = 0.83 * rain_24h - 1.27
            q_0 = 800.0 * np.exp(-prev_dc / 400.0)
            q_r = q_0 + 3.937 * r_d
            prev_dc = max(0.0, 400.0 * np.log(800.0 / max(1.0, q_r)))

        t_k = max(-2.8, temp)
        v = 0.36 * (t_k + 2.8) + l_f
        return float(max(0.0, prev_dc + 0.5 * max(0.0, v)))

    def _calc_isi(self, wind_kmh: float, ffmc: float) -> float:
        f_w = np.exp(0.05039 * wind_kmh)
        m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        f_f = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
        return float(0.208 * f_w * f_f)

    def _calc_bui(self, dmc: float, dc: float) -> float:
        if dmc <= 0.0 and dc <= 0.0:
            return 0.0

        denom = dmc + 0.4 * dc
        if denom <= 1e-6:
            return 0.0

        if dmc <= 0.4 * dc:
            bui = 0.8 * dmc * dc / denom
        else:
            bui = dmc - (1.0 - 0.8 * dc / denom) * (0.92 + (0.0114 * dmc) ** 1.7)
        return float(max(0.0, bui))

    def _calc_fwi(self, isi: float, bui: float) -> float:
        if bui <= 80.0:
            f_d = 0.626 * (bui ** 0.80) + 0.1
        else:
            f_d = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))

        b = 0.1 * isi * f_d
        if b > 1.0:
            ln_s = 2.72 * ((0.434 * np.log(b)) ** 0.647)
            return float(np.exp(ln_s))
        return float(b)

    def _step(self, state: Dict[str, float], record: Dict[str, Any], lat: float) -> Dict[str, float]:
        l_e, l_f = self._get_day_length_factors(lat, record["month"])

        ffmc = self._calc_ffmc(
            temp=record["temp_noon"],
            rh=record["rh_noon"],
            wind_kmh=record["wind_noon_kmh"],
            rain_24h=record["rain_24h_mm"],
            prev_ffmc=state["FFMC"],
        )
        dmc = self._calc_dmc(
            temp=record["temp_noon"],
            rh=record["rh_noon"],
            rain_24h=record["rain_24h_mm"],
            l_e=l_e,
            prev_dmc=state["DMC"],
        )
        dc = self._calc_dc(
            temp=record["temp_noon"],
            rain_24h=record["rain_24h_mm"],
            l_f=l_f,
            prev_dc=state["DC"],
        )

        isi = self._calc_isi(wind_kmh=record["wind_noon_kmh"], ffmc=ffmc)
        bui = self._calc_bui(dmc=dmc, dc=dc)
        fwi = self._calc_fwi(isi=isi, bui=bui)

        return {
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui,
            "FWI": fwi,
        }


if __name__ == "__main__":
    fetcher = CFFDRSFetcher(spinup_days=90)

    # Test globally diverse wildfire-prone regions
    GLOBAL_TEST_SITES = {
        "Canada (Alberta Boreal)": (56.7264, -111.3803, "2023-07-15"),
        "Australia (NSW Bushfire season)": (-33.8688, 151.2093, "2020-01-05"),
        "Greece (Mediterranean summer)": (38.0408, 23.8202, "2023-08-22"),
        "Indonesia (Equatorial peatland)": (-0.7893, 113.9213, "2023-09-15"),
    }

    for name, (lat, lon, target_date) in GLOBAL_TEST_SITES.items():
        print(f"\n{'='*60}")
        print(f"Site: {name} [{lat}, {lon}] at {target_date}")
        metrics = fetcher.fetch_cffdrs_metrics(lat, lon, date_t0=target_date)
        for k, v in metrics.items():
            print(f"  {k:<16}: {v:6.2f}")