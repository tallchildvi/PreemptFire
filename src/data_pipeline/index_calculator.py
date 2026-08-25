from typing import Optional
import numpy as np


class IndexCalculator:

    def __init__(
        self,
        epsilon: float = 1e-7,
        reflectance_scale: float = 10000.0,
        auto_detect_scale: bool = False,
        sar_ratio_max: float = 10.0,
    ):
        self.epsilon = float(epsilon)
        self.reflectance_scale = float(reflectance_scale)
        self.auto_detect_scale = bool(auto_detect_scale)
        self.sar_ratio_max = float(sar_ratio_max)

    
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
        results: dict[str, np.ndarray] = {}

        # 1. T0 Optical & Fuel Moisture Indices
        results["NDVI_T0"] = self.calc_ndvi(b08_t0, b04_t0)
        results["NDMI_T0"] = self.calc_ndmi(b08_t0, b11_t0)
        results["MSI_T0"] = self.calc_msi(b11_t0, b08_t0)
        results["NBR_T0"] = self.calc_nbr(b08_t0, b12_t0)
        results["NBR2_T0"] = self.calc_nbr2(b11_t0, b12_t0)
        results["NMDI_T0"] = self.calc_nmdi(b08_t0, b11_t0, b12_t0)
        results["EVI_T0"] = self.calc_evi(b08_t0, b04_t0, b02_t0)

        if b05_t0 is not None:
            results["NDRE_T0"] = self.calc_ndre(b8a_t0, b05_t0)

        # 2. Quality & Physical State Masks
        results["MASK_SNOW"] = self.calc_snow_mask(scl_t0)
        results["MASK_OPTICAL_INVALID"] = self.calc_optical_invalid_mask(scl_t0)
        results["MASK_WATER"] = self.calc_water_mask(scl_t0)

        # 3. Temporal Spectral Changes (T0 - Tprev)
        if b08_tprev is not None and b04_tprev is not None:
            results["dNDVI"] = self.calc_delta(results["NDVI_T0"], self.calc_ndvi(b08_tprev, b04_tprev))

        if b08_tprev is not None and b11_tprev is not None:
            results["dNDMI"] = self.calc_delta(results["NDMI_T0"], self.calc_ndmi(b08_tprev, b11_tprev))

        if b08_tprev is not None and b12_tprev is not None:
            results["dNBR"] = self.calc_delta(results["NBR_T0"], self.calc_nbr(b08_tprev, b12_tprev))

        # 4. Sentinel-1 SAR Polarimetric Features
        if sar_vv is not None and sar_vh is not None:
            results["SAR_RATIO"] = self.calc_sar_ratio(sar_vh, sar_vv)
            results["SAR_RVI"] = self.calc_sar_rvi(sar_vh, sar_vv)

        return results

    
    # Common Utilities
    

    def _calc_norm_diff(self, band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
        denom = band_a + band_b
        valid = np.isfinite(band_a) & np.isfinite(band_b) & (np.abs(denom) > self.epsilon)

        out = np.zeros_like(band_a, dtype=np.float32)
        np.divide(band_a - band_b, denom, out=out, where=valid)
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    
    # Vegetation & Moisture Indices
    

    def calc_ndvi(self, b08_nir: np.ndarray, b04_red: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b08_nir, b04_red)

    def calc_ndre(self, b8a_narrow_nir: np.ndarray, b05_rededge: np.ndarray) -> np.ndarray:
        """Normalized Difference Red Edge Index using Sentinel-2 B8A (~865 nm) and B05 (~705 nm)."""
        return self._calc_norm_diff(b8a_narrow_nir, b05_rededge)

    def calc_ndmi(self, b08_nir: np.ndarray, b11_swir1: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b08_nir, b11_swir1)

    def calc_nbr(self, b08_nir: np.ndarray, b12_swir2: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b08_nir, b12_swir2)

    def calc_nbr2(self, b11_swir1: np.ndarray, b12_swir2: np.ndarray) -> np.ndarray:
        return self._calc_norm_diff(b11_swir1, b12_swir2)

    def calc_msi(self, b11_swir1: np.ndarray, b08_nir: np.ndarray) -> np.ndarray:
        valid = np.isfinite(b11_swir1) & np.isfinite(b08_nir) & (b08_nir > self.epsilon)
        out = np.zeros_like(b11_swir1, dtype=np.float32)
        np.divide(b11_swir1, b08_nir, out=out, where=valid)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0).astype(np.float32)

    def calc_nmdi(self, b08_nir: np.ndarray, b11_swir1: np.ndarray, b12_swir2: np.ndarray) -> np.ndarray:
        swir_diff = b11_swir1 - b12_swir2
        numerator = b08_nir - swir_diff
        denominator = b08_nir + swir_diff

        valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > self.epsilon)
        out = np.zeros_like(b08_nir, dtype=np.float32)
        np.divide(numerator, denominator, out=out, where=valid)
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    
    # EVI
    

    def _get_reflectance_scale(self, reference_band: np.ndarray) -> float:
        if not self.auto_detect_scale:
            return self.reflectance_scale

        finite = reference_band[np.isfinite(reference_band)]
        if finite.size == 0:
            return 1.0

        p95 = np.percentile(finite, 95)
        return self.reflectance_scale if p95 > 2.0 else 1.0

    def calc_evi(self, b08_nir: np.ndarray, b04_red: np.ndarray, b02_blue: np.ndarray) -> np.ndarray:
        scale = self._get_reflectance_scale(b08_nir)
        nir = b08_nir / scale
        red = b04_red / scale
        blue = b02_blue / scale

        denominator = nir + 6.0 * red - 7.5 * blue + 1.0
        valid = np.isfinite(nir) & np.isfinite(red) & np.isfinite(blue) & (denominator > self.epsilon)

        out = np.zeros_like(nir, dtype=np.float32)
        np.divide(2.5 * (nir - red), denominator, out=out, where=valid)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)

    
    # Sentinel-2 Quality Masks (Boolean Arrays)
    

    def calc_snow_mask(self, scl: np.ndarray) -> np.ndarray:
        """True where Sentinel-2 SCL identifies Snow/Ice (SCL = 11)."""
        return scl == 11

    def calc_optical_invalid_mask(self, scl: np.ndarray) -> np.ndarray:
        """True where optical observations are unreliable (SCL: 2, 3, 8, 9, 10)."""
        return (scl == 2) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)

    def calc_water_mask(self, scl: np.ndarray) -> np.ndarray:
        """True where Sentinel-2 SCL identifies water (SCL = 6)."""
        return scl == 6

    
    # Temporal Changes
    

    def calc_delta(self, index_t0: np.ndarray, index_tprev: np.ndarray) -> np.ndarray:
        """Temporal difference: index(T0) - index(Tprev), bounded to [-2, 2]."""
        delta = index_t0.astype(np.float32) - index_tprev.astype(np.float32)
        return np.clip(np.nan_to_num(delta, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0).astype(np.float32)

    
    # Sentinel-1 SAR Features
    

    def calc_sar_ratio(self, vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
        """VH/VV cross-polarization ratio in linear power scale."""
        valid = np.isfinite(vh) & np.isfinite(vv) & (vv > self.epsilon)
        out = np.zeros_like(vh, dtype=np.float32)
        np.divide(vh, vv, out=out, where=valid)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=self.sar_ratio_max, neginf=0.0), 0.0, self.sar_ratio_max).astype(np.float32)

    def calc_sar_rvi(self, vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
        """Dual-polarization Radar Vegetation Index: 4*VH / (VV + VH) in linear power scale."""
        denominator = vv + vh
        valid = np.isfinite(vh) & np.isfinite(vv) & (denominator > self.epsilon)
        out = np.zeros_like(vh, dtype=np.float32)
        np.divide(4.0 * vh, denominator, out=out, where=valid)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=4.0, neginf=0.0), 0.0, 4.0).astype(np.float32)