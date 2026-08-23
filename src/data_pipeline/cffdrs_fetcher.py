from datetime import datetime, timedelta
from typing import Any, Dict, List
import numpy as np
import requests


class CFFDRSFetcher:
    """
    Autonomous fetcher and computer for the Canadian Forest Fire Danger Rating System (CFFDRS).
    Executes a single 90-day historical weather request via Open-Meteo API, extracts strict 
    12:00 LST (solar noon) records with 24h antecedent rainfall, and computes fuel moisture 
    codes (FFMC, DMC, DC) and fire behavior indices (ISI, BUI, FWI) at T0.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Empirical effective day-length factors by month (Jan -> Dec) for northern latitudes (>= 45°N)
    DMC_DAY_LENGTH: List[float] = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    DC_DAY_LENGTH: List[float] = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

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
        and returns all 6 CFFDRS indices calibrated for date_t0.
        """
        d_end = datetime.strptime(date_t0, "%Y-%m-%d")
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
            state = self._step(state, record)

        return {
            "ffmc": float(state["FFMC"]),
            "dmc": float(state["DMC"]),
            "dc": float(state["DC"]),
            "isi": float(state["ISI"]),
            "bui": float(state["BUI"]),
            "fwi": float(state["FWI"]),
        }

    
    # Private Ingestion & Alignment Helpers
    

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

    def _extract_noon_records(self, hourly: dict, lon: float) -> List[Dict[str, Any]]:
        offset_hours = int(round(lon / 15.0))
        times_utc = [datetime.fromisoformat(t) for t in hourly["time"]]
        times_lst = [t + timedelta(hours=offset_hours) for t in times_utc]

        temps = np.array(hourly["temperature_2m"], dtype=np.float32)
        rhs = np.array(hourly["relative_humidity_2m"], dtype=np.float32)
        winds = np.array(hourly["wind_speed_10m"], dtype=np.float32)
        precips = np.array(hourly["precipitation"], dtype=np.float32)

        noon_records: List[Dict[str, Any]] = []
        n_hours = len(times_lst)

        for i in range(23, n_hours):
            cur_time = times_lst[i]
            if cur_time.hour == 12:
                # 24h precipitation: sum from 13:00 yesterday to 12:00 today
                rain_24h = float(np.nansum(precips[i - 23 : i + 1]))

                noon_records.append({
                    "date": cur_time.date(),
                    "month": cur_time.month,
                    "temp_noon": float(temps[i]),
                    "rh_noon": float(np.clip(rhs[i], 1.0, 100.0)),
                    "wind_noon_kmh": max(0.0, float(winds[i])),
                    "rain_24h_mm": max(0.0, rain_24h),
                })

        return noon_records

    
    # Private Physical Formulations (Van Wagner, 1987)
    

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

    def _calc_dmc(self, temp: float, rh: float, rain_24h: float, month: int, prev_dmc: float) -> float:
        l_e = self.DMC_DAY_LENGTH[int(np.clip(month, 1, 12)) - 1]

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

    def _calc_dc(self, temp: float, rain_24h: float, month: int, prev_dc: float) -> float:
        l_f = self.DC_DAY_LENGTH[int(np.clip(month, 1, 12)) - 1]

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

    def _step(self, state: Dict[str, float], record: Dict[str, Any]) -> Dict[str, float]:
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
            month=record["month"],
            prev_dmc=state["DMC"],
        )
        dc = self._calc_dc(
            temp=record["temp_noon"],
            rain_24h=record["rain_24h_mm"],
            month=record["month"],
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
    # Test on Fort McMurray during wildfire season
    lat, lon = 56.7264, -111.3803
    date_t0 = "2023-07-15"

    fetcher = CFFDRSFetcher(spinup_days=90)
    cffdrs_metrics = fetcher.fetch_cffdrs_metrics(lat, lon, date_t0=date_t0)

    print(f"\n--- CFFDRS Indices at T0 [{lat}, {lon}] ({date_t0}) ---")
    for k, v in cffdrs_metrics.items():
        print(f"  {k:<16}: {v:6.2f}")