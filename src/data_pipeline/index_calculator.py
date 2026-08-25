from typing import Optional
import numpy as np


class IndexCalculator:
    """
    Production-hardened 2D raster index processor.
    Calculates optical vegetation & fuel moisture indices, temporal deltas,
    Sentinel-1 radar polarimetric features, and quality boolean masks.
    Compatible with Numba JIT and strictly typed array pipelines.
    """

    def __init__(
        self,
        epsilon: float = 1e-7,
        reflectance_scale: float = 10000.0,
        auto_detect_scale: bool = True,
    ):
        self.epsilon = float(epsilon)
        self.reflectance_scale = float(reflectance_scale)
        self.auto_detect_scale = auto_detect_scale

    
    # Main Orchestrator
    

    def compute_all_indices(
        self,
        b02_t0: np.ndarray,
        b04_t0: np.ndarray,
        b08_t0: np.ndarray,
        b8a_t0: np.ndarray,
        b11_t0: np.ndarray,
        b12_t0: np.ndarray,
        scl_t0: np.ndarray,
        b05_t0: Optional[np.ndarray] = None,
        b04_tprev: Optional[np.ndarray] = None,
        b08_tprev: Optional[np.ndarray] = None,
        b11_tprev: Optional[np.ndarray] = None,
        b12_tprev: Optional[np.ndarray] = None,
        sar_vv: Optional[np.ndarray] = None,
        sar_vh: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Orchestrates full 2D spatial raster generation from explicit array inputs.
        """
        results: dict[str, np.ndarray] = {}

        # 1. T0 Optical & Fuel Moisture Indices
        results["NDVI_T0"] = self.calc_ndvi(b08_t0, b04_t0)
        results["NDMI_T0"] = self.calc_ndmi(b08_t0, b11_t0)
        results["MSI_T0"] = self.calc_msi(b11_t0, b08_t0)
        results["NBR_T0"] = self.calc_nbr(b08_t0, b12_t0)
        results["NBR2_T0"] = self.calc_nbr2(b11_t0, b12_t0)
        results["NMDI_T0"] = self.calc_nmdi(b08_t0, b11_t0, b12_t0)
        results["EVI_T0"] = self.calc_evi(b08_t0, b04_t0, b02_t0)

        # Narrow-band Red Edge Index
        if b05_t0 is not None:
            results["NDRE_T0"] = self.calc_ndre(b8a_t0, b05_t0)

        # 2. Quality & Physical State Masks (Boolean arrays)
        results["MASK_SNOW"] = self.calc_snow_mask(scl_t0)
        results["MASK_CLOUD_SHADOW"] = self.calc_cloud_shadow_mask(scl_t0)
        results["MASK_WATER"] = self.calc_water_mask(scl_t0)

        # 3. Temporal Drying Trends (T0 - Tprev)
        if b08_tprev is not None and b04_tprev is not None:
            results["dNDVI"] = self.calc_delta(results["NDVI_T0"], self.calc_ndvi(b08_tprev, b04_tprev))

        if b08_tprev is not None and b11_tprev is not None:
            results["dNDMI"] = self.calc_delta(results["NDMI_T0"], self.calc_ndmi(b08_tprev, b11_tprev))

        if b08_tprev is not None and b12_tprev is not None:
            results["dNBR"] = self.calc_delta(results["NBR_T0"], self.calc_nbr(b08_tprev, b12_tprev))

        # 4. Sentinel-1 SAR Polarimetric Metrics
        if sar_vv is not None and sar_vh is not None:
            results["SAR_RATIO"] = self.calc_sar_ratio(sar_vh, sar_vv)
            results["SAR_RVI"] = self.calc_sar_rvi(sar_vh, sar_vv)

        return results

    
    # Core Mathematical Formulations
    

    def _calc_norm_diff(self, band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
        denom = band_a + band_b
        out = np.where(np.abs(denom) > self.epsilon, (band_a - band_b) / denom, 0.0)
        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0).astype(np.float32)

    def calc_ndvi(self, b08_nir: np.ndarray, b04_red: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b08_nir, b04_red)

    def calc_ndre(self, b8a_narrow_nir: np.ndarray, b05_rededge: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b8a_narrow_nir, b05_rededge)

    def calc_ndmi(self, b08_nir: np.ndarray, b11_swir1: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b08_nir, b11_swir1)

    def calc_nbr(self, b08_nir: np.ndarray, b12_swir2: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b08_nir, b12_swir2)

    def calc_nbr2(self, b11_swir1: np.ndarray, b12_swir2: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b11_swir1, b12_swir2)

    def calc_msi(self, b11_swir1: np.ndarray, b08_nir: np.ndarray) -> np.ndarray:
        out = np.where(b08_nir > self.epsilon, b11_swir1 / b08_nir, 0.0)
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 5.0).astype(np.float32)

    def calc_nmdi(self, b08_nir: np.ndarray, b11_swir1: np.ndarray, b12_swir2: np.ndarray) -> np.ndarray:
        swir_diff = b11_swir1 - b12_swir2
        num = b08_nir - swir_diff
        denom = b08_nir + swir_diff
        out = np.where(np.abs(denom) > self.epsilon, num / denom, 0.0)
        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0).astype(np.float32)

    def calc_evi(self, b08_nir: np.ndarray, b04_red: np.ndarray, b02_blue: np.ndarray) -> np.ndarray:
        scale = 1.0
        if self.auto_detect_scale:
            # Check 95th percentile to avoid false positives on clouds/snow outliers
            p95 = np.nanpercentile(b08_nir, 95)
            if p95 > 2.0:
                scale = self.reflectance_scale
        else:
            scale = self.reflectance_scale

        nir = b08_nir / scale
        red = b04_red / scale
        blue = b02_blue / scale

        denom = nir + 6.0 * red - 7.5 * blue + 1.0
        out = np.where(denom > self.epsilon, 2.5 * (nir - red) / denom, 0.0)
        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0).astype(np.float32)

    
    # Quality & State Masks (Boolean)
    
    def calc_snow_mask(self, scl: np.ndarray) -> np.ndarray:
        """True where Snow / Ice is detected (SCL 11)."""
        return scl == 11

    def calc_cloud_shadow_mask(self, scl: np.ndarray) -> np.ndarray:
        """
        True for invalid optical pixels:
        SCL 2: Dark area pixels (shadow / low illumination)
        SCL 3: Cloud shadows
        SCL 8: Cloud medium probability
        SCL 9: Cloud high probability
        SCL 10: Thin cirrus
        """
        return (scl == 2) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)

    def calc_water_mask(self, scl: np.ndarray) -> np.ndarray:
        """True for water bodies."""
        return scl == 6

    
    # Temporal Deltas & SAR Polarimetry
    

    def calc_delta(self, index_t0: np.ndarray, index_tprev: np.ndarray) -> np.ndarray:
        """Clamped temporal delta: [-2.0, 2.0]."""
        delta = index_t0 - index_tprev
        return np.clip(np.nan_to_num(delta, nan=0.0), -2.0, 2.0).astype(np.float32)

    def calc_sar_ratio(self, vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
        """SAR Cross-Ratio (VH / VV) in linear scale."""
        ratio = np.where(vv > self.epsilon, vh / vv, 0.0)
        return np.maximum(0.0, np.nan_to_num(ratio, nan=0.0)).astype(np.float32)

    def calc_sar_rvi(self, vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
        denom = vv + vh
        rvi = np.where(denom > self.epsilon, (4.0 * vh) / denom, 0.0)

        return np.clip(np.nan_to_num(rvi, nan=0.0, posinf=4.0, neginf=0.0), 0.0, 4.0).astype(np.float32)