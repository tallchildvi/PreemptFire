from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CFFDRSEngine:
    """
    Object-oriented Canadian Forest Fire Danger Rating System (CFFDRS) engine.
    Implements standard Van Wagner (1987) formulations for fuel moisture codes.
    """

    # Empirical effective day-length factors by month (Jan -> Dec)
    DMC_DAY_LENGTH: List[float] = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    DC_DAY_LENGTH: List[float] = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

    FEATURE_NAMES: List[str] = ["cffdrs_ffmc", "cffdrs_dmc", "cffdrs_dc", "cffdrs_isi", "cffdrs_bui", "cffdrs_fwi"]

    def __init__(
        self,
        default_ffmc: float = 85.0,
        default_dmc: float = 6.0,
        default_dc: float = 15.0,
        track_history: bool = False,
    ):
        self.default_ffmc = float(default_ffmc)
        self.default_dmc = float(default_dmc)
        self.default_dc = float(default_dc)
        self.track_history = track_history

        self.current_state: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []
        self.reset_state()

    def reset_state(self) -> None:
        """Resets engine state to default spring baseline conditions."""
        self.current_state = {
            "FFMC": self.default_ffmc,
            "DMC": self.default_dmc,
            "DC": self.default_dc,
            "ISI": 0.0,
            "BUI": 0.0,
            "FWI": 0.0,
        }
        self.history.clear()

    
    # Moisture Codes (FFMC, DMC, DC)


    def calculate_ffmc(
        self,
        temp: float,
        rh: float,
        wind_kmh: float,
        rain_24h: float,
        prev_ffmc: Optional[float] = None,
    ) -> float:
        """Fine Fuel Moisture Code (FFMC) for top 1.2 cm surface litter/cured grass."""
        p_ffmc = self.current_state["FFMC"] if prev_ffmc is None else prev_ffmc
        rh_safe = float(np.clip(rh, 1.0, 100.0))
        wind_safe = max(0.0, float(wind_kmh))
        rain_safe = max(0.0, float(rain_24h))

        m_0 = 147.2 * (101.0 - p_ffmc) / (59.5 + p_ffmc)

        # Rain routine
        if rain_safe > 0.5:
            r_a = rain_safe - 0.5
            if m_0 > 150.0:
                m_r = (
                    m_0
                    + 42.5 * r_a * np.exp(-100.0 / (251.0 - m_0)) * (1.0 - np.exp(-6.93 / r_a))
                    + 0.0015 * ((m_0 - 150.0) ** 2) * np.sqrt(r_a)
                )
            else:
                m_r = m_0 + 42.5 * r_a * np.exp(-100.0 / (251.0 - m_0)) * (1.0 - np.exp(-6.93 / r_a))
            m_0 = min(250.0, m_r)

        # Equilibrium moisture content for drying (E_d) and wetting (E_w)
        e_d = (
            0.942 * (rh_safe ** 0.679)
            + 11.0 * np.exp((rh_safe - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh_safe))
        )
        e_w = (
            0.618 * (rh_safe ** 0.753)
            + 10.0 * np.exp((rh_safe - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh_safe))
        )

        # Moisture step calculation
        if m_0 > e_d:
            k_a = 0.424 * (1.0 - (rh_safe / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_safe) * (1.0 - (rh_safe / 100.0) ** 8)
            k_d = k_a * 0.581 * np.exp(0.0365 * temp)
            m = e_d + (m_0 - e_d) * (10.0 ** (-k_d))
        elif m_0 < e_w:
            k_b = 0.424 * (1.0 - ((100.0 - rh_safe) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_safe) * (
                1.0 - ((100.0 - rh_safe) / 100.0) ** 8
            )
            k_w = k_b * 0.581 * np.exp(0.0365 * temp)
            m = e_w - (e_w - m_0) * (10.0 ** (-k_w))
        else:
            m = m_0

        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        return float(np.clip(ffmc, 0.0, 101.0))

    def calculate_dmc(
        self,
        temp: float,
        rh: float,
        rain_24h: float,
        month: int,
        prev_dmc: Optional[float] = None,
    ) -> float:
        """Duff Moisture Code (DMC) for loosely compacted duff layer (5-10 cm)."""
        p_dmc = self.current_state["DMC"] if prev_dmc is None else prev_dmc
        m_idx = int(np.clip(month, 1, 12)) - 1
        l_e = self.DMC_DAY_LENGTH[m_idx]

        rh_safe = float(np.clip(rh, 1.0, 100.0))
        rain_safe = max(0.0, float(rain_24h))

        # Rain routine
        if rain_safe > 1.5:
            r_e = 0.92 * rain_safe - 1.27
            m_0 = 20.0 + np.exp(5.6348 - p_dmc / 43.43)
            if p_dmc <= 33.0:
                b = 100.0 / (0.5 + 0.3 * p_dmc)
            elif p_dmc <= 65.0:
                b = 14.0 - 1.3 * np.log(p_dmc)
            else:
                b = 6.2 * np.log(p_dmc) - 17.2

            m_r = m_0 + 1000.0 * r_e / (48.77 + b * r_e)
            p_dmc = max(0.0, 244.72 - 43.43 * np.log(max(1.0, m_r - 20.0)))

        # Evaporative drying factor
        t_k = max(-1.1, temp)
        k = 1.894 * (t_k + 1.1) * (100.0 - rh_safe) * l_e * 1e-4
        return float(max(0.0, p_dmc + max(0.0, k)))

    def calculate_dc(
        self,
        temp: float,
        rain_24h: float,
        month: int,
        prev_dc: Optional[float] = None,
    ) -> float:
        """Drought Code (DC) for deep, compact organic layers and peat (10-20+ cm)."""
        p_dc = self.current_state["DC"] if prev_dc is None else prev_dc
        m_idx = int(np.clip(month, 1, 12)) - 1
        l_f = self.DC_DAY_LENGTH[m_idx]
        rain_safe = max(0.0, float(rain_24h))

        # Rain routine
        if rain_safe > 2.8:
            r_d = 0.83 * rain_safe - 1.27
            q_0 = 800.0 * np.exp(-p_dc / 400.0)
            q_r = q_0 + 3.937 * r_d
            p_dc = max(0.0, 400.0 * np.log(800.0 / max(1.0, q_r)))

        # Evaporative drying
        t_k = max(-2.8, temp)
        v = 0.36 * (t_k + 2.8) + l_f
        return float(max(0.0, p_dc + 0.5 * max(0.0, v)))

    
    # Fire Behavior Indices (ISI, BUI, FWI)
    

    def calculate_isi(self, wind_kmh: float, ffmc: Optional[float] = None) -> float:
        """Initial Spread Index (ISI) combining fine fuel dryness and wind rate of spread."""
        cur_ffmc = self.current_state["FFMC"] if ffmc is None else ffmc
        wind_safe = max(0.0, float(wind_kmh))

        f_w = np.exp(0.05039 * wind_safe)
        m = 147.2 * (101.0 - cur_ffmc) / (59.5 + cur_ffmc)
        f_f = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
        return float(0.208 * f_w * f_f)

    def calculate_bui(self, dmc: Optional[float] = None, dc: Optional[float] = None) -> float:
        """Buildup Index (BUI) combining available medium and deep fuel volume."""
        cur_dmc = self.current_state["DMC"] if dmc is None else dmc
        cur_dc = self.current_state["DC"] if dc is None else dc

        if cur_dmc <= 0.0 and cur_dc <= 0.0:
            return 0.0

        denom = cur_dmc + 0.4 * cur_dc
        if denom <= 1e-6:
            return 0.0

        if cur_dmc <= 0.4 * cur_dc:
            bui = 0.8 * cur_dmc * cur_dc / denom
        else:
            bui = cur_dmc - (1.0 - 0.8 * cur_dc / denom) * (0.92 + (0.0114 * cur_dmc) ** 1.7)
        return float(max(0.0, bui))

    def calculate_fwi(self, isi: Optional[float] = None, bui: Optional[float] = None) -> float:
        """Fire Weather Index (FWI) representing frontal fire intensity (kW/m)."""
        cur_isi = self.current_state["ISI"] if isi is None else isi
        cur_bui = self.current_state["BUI"] if bui is None else bui

        if cur_bui <= 80.0:
            f_d = 0.626 * (cur_bui ** 0.80) + 0.1
        else:
            f_d = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * cur_bui))

        b = 0.1 * cur_isi * f_d
        if b > 1.0:
            ln_s = 2.72 * ((0.434 * np.log(b)) ** 0.647)
            return float(np.exp(ln_s))
        return float(b)

    
    # Pipeline Step & Spin-Up Handlers

    def step(
        self,
        temp_noon: float,
        rh_noon: float,
        wind_noon_kmh: float,
        rain_24h_mm: float,
        month: int,
    ) -> Dict[str, float]:
        """Advances internal engine state by 1 daily noon-to-noon step."""
        ffmc = self.calculate_ffmc(temp_noon, rh_noon, wind_noon_kmh, rain_24h_mm)
        dmc = self.calculate_dmc(temp_noon, rh_noon, rain_24h_mm, month)
        dc = self.calculate_dc(temp_noon, rain_24h_mm, month)

        isi = self.calculate_isi(wind_noon_kmh, ffmc=ffmc)
        bui = self.calculate_bui(dmc=dmc, dc=dc)
        fwi = self.calculate_fwi(isi=isi, bui=bui)

        self.current_state = {
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui,
            "FWI": fwi,
        }

        if self.track_history:
            self.history.append(self.current_state.copy())

        return self.current_state

    def process_spinup_series(self, noon_records: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Processes a multi-day sequential spin-up history up to T0.
        Returns the calibrated 6-variable dictionary on the target observation day.
        """
        self.reset_state()
        for record in noon_records:
            self.step(
                temp_noon=record["temp_noon"],
                rh_noon=record["rh_noon"],
                wind_noon_kmh=record["wind_noon_kmh"],
                rain_24h_mm=record["rain_24h_mm"],
                month=record["month"],
            )
        return self.current_state

    def get_state_vector(self) -> np.ndarray:
        """Returns the current 6-variable state as a 1D float32 NumPy array."""
        return np.array(
            [
                self.current_state["FFMC"],
                self.current_state["DMC"],
                self.current_state["DC"],
                self.current_state["ISI"],
                self.current_state["BUI"],
                self.current_state["FWI"],
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":
    engine = CFFDRSEngine(track_history=True)

    # Synthetic 5-day heatwave test sequence in July
    test_sequence = [
        {"temp_noon": 22.0, "rh_noon": 45.0, "wind_noon_kmh": 15.0, "rain_24h_mm": 0.0, "month": 7},
        {"temp_noon": 25.5, "rh_noon": 35.0, "wind_noon_kmh": 18.0, "rain_24h_mm": 0.0, "month": 7},
        {"temp_noon": 28.0, "rh_noon": 28.0, "wind_noon_kmh": 22.0, "rain_24h_mm": 0.0, "month": 7},
        {"temp_noon": 31.0, "rh_noon": 20.0, "wind_noon_kmh": 26.0, "rain_24h_mm": 0.0, "month": 7},
        {"temp_noon": 33.5, "rh_noon": 14.0, "wind_noon_kmh": 34.0, "rain_24h_mm": 0.0, "month": 7},
    ]

    target_state = engine.process_spinup_series(test_sequence)
    vector = engine.get_state_vector()

    print("\n--- Final CFFDRS State at T0 ---")
    for key, val in target_state.items():
        print(f"  {key:<6}: {val:6.2f}")