import numpy as np


class IndexCalculator:
    """Calculates multi-spectral optical and SAR polarimetric indices."""

    @staticmethod
    def calc_delta(t0_arr: np.ndarray, tprev_arr: np.ndarray) -> np.ndarray:
        """Computes temporal difference between t0 and tprev rasters."""
        return (t0_arr - tprev_arr).astype(np.float32)

    @staticmethod
    def calc_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Vegetation Index: (B08 - B04) / (B08 + B04)."""
        return ((nir - red) / (nir + red + eps)).astype(np.float32)

    @staticmethod
    def calc_ndmi(nir: np.ndarray, swir1: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Moisture Index: (B08 - B11) / (B08 + B11)."""
        return ((nir - swir1) / (nir + swir1 + eps)).astype(np.float32)

    @staticmethod
    def calc_nbr(nir: np.ndarray, swir2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Burn Ratio: (B08 - B12) / (B08 + B12)."""
        return ((nir - swir2) / (nir + swir2 + eps)).astype(np.float32)

    @staticmethod
    def calc_nbr2(swir1: np.ndarray, swir2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Burn Ratio 2: (B11 - B12) / (B11 + B12)."""
        return ((swir1 - swir2) / (swir1 + swir2 + eps)).astype(np.float32)

    @staticmethod
    def calc_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Enhanced Vegetation Index: 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)."""
        numerator = 2.5 * (nir - red)
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0 + eps
        return (numerator / denominator).astype(np.float32)

    @staticmethod
    def calc_ndre(nir: np.ndarray, red_edge: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Normalized Difference Red Edge Index: (B08 - B05) / (B08 + B05)."""
        return ((nir - red_edge) / (nir + red_edge + eps)).astype(np.float32)

    @staticmethod
    def calc_msi(swir1: np.ndarray, nir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Moisture Stress Index: B11 / B08."""
        return ((swir1 + eps) / (nir + eps)).astype(np.float32)

    @staticmethod
    def calc_nmdi(
        nir: np.ndarray,
        swir1: np.ndarray,
        swir2: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Normalized Multi-band Drought Index: (B08 - (B11 - B12)) / (B08 + (B11 - B12))."""
        diff_swir = swir1 - swir2
        numerator = nir - diff_swir
        denominator = nir + diff_swir + eps
        return (numerator / denominator).astype(np.float32)

    @staticmethod
    def calc_sar_ratio(vh: np.ndarray, vv: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Cross-polarization ratio: VH / VV."""
        return ((vh + eps) / (vv + eps)).astype(np.float32)

    @staticmethod
    def calc_sar_rvi(vv: np.ndarray, vh: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Radar Vegetation Index: 4 * VH / (VV + VH)."""
        return ((4.0 * vh) / (vv + vh + eps)).astype(np.float32)

    def compute_all_indices(
        self,
        b02: np.ndarray | None = None,
        b03: np.ndarray | None = None,
        b04: np.ndarray | None = None,
        b05: np.ndarray | None = None,
        b06: np.ndarray | None = None,
        b07: np.ndarray | None = None,
        b08: np.ndarray | None = None,
        b8a: np.ndarray | None = None,
        b11: np.ndarray | None = None,
        b12: np.ndarray | None = None,
        scl: np.ndarray | None = None,
        sar_vv: np.ndarray | None = None,
        sar_vh: np.ndarray | None = None,
        b04_tprev: np.ndarray | None = None,
        b08_tprev: np.ndarray | None = None,
        b11_tprev: np.ndarray | None = None,
        b12_tprev: np.ndarray | None = None,
        b02_t0: np.ndarray | None = None,
        b03_t0: np.ndarray | None = None,
        b04_t0: np.ndarray | None = None,
        b05_t0: np.ndarray | None = None,
        b06_t0: np.ndarray | None = None,
        b07_t0: np.ndarray | None = None,
        b08_t0: np.ndarray | None = None,
        b8a_t0: np.ndarray | None = None,
        b11_t0: np.ndarray | None = None,
        b12_t0: np.ndarray | None = None,
        scl_t0: np.ndarray | None = None,
        sar_vv_t0: np.ndarray | None = None,
        sar_vh_t0: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Computes static T0 indices and temporal deltas with support for both band naming styles."""
        b02 = b02 if b02 is not None else b02_t0
        b04 = b04 if b04 is not None else b04_t0
        b05 = b05 if b05 is not None else b05_t0
        b08 = b08 if b08 is not None else b08_t0
        b11 = b11 if b11 is not None else b11_t0
        b12 = b12 if b12 is not None else b12_t0
        sar_vv = sar_vv if sar_vv is not None else sar_vv_t0
        sar_vh = sar_vh if sar_vh is not None else sar_vh_t0

        results: dict[str, np.ndarray] = {}

        # 1. Optical Indices (T0)
        if b08 is not None and b04 is not None:
            results["NDVI_T0"] = self.calc_ndvi(b08, b04)
        if b08 is not None and b11 is not None:
            results["NDMI_T0"] = self.calc_ndmi(b08, b11)
            results["MSI_T0"] = self.calc_msi(b11, b08)
        if b08 is not None and b12 is not None:
            results["NBR_T0"] = self.calc_nbr(b08, b12)
        if b11 is not None and b12 is not None:
            results["NBR2_T0"] = self.calc_nbr2(b11, b12)
        if b08 is not None and b04 is not None and b02 is not None:
            results["EVI_T0"] = self.calc_evi(b08, b04, b02)
        if b08 is not None and b05 is not None:
            results["NDRE_T0"] = self.calc_ndre(b08, b05)
        if b08 is not None and b11 is not None and b12 is not None:
            results["NMDI_T0"] = self.calc_nmdi(b08, b11, b12)

        # 2. SAR Polarimetric Indices
        if sar_vv is not None and sar_vh is not None:
            results["SAR_RATIO"] = self.calc_sar_ratio(sar_vh, sar_vv)
            results["SAR_RVI"] = self.calc_sar_rvi(sar_vv, sar_vh)

        # 3. Temporal Spectral Changes (T0 - Tprev)
        if b08_tprev is not None and b04_tprev is not None and "NDVI_T0" in results:
            results["dNDVI"] = self.calc_delta(results["NDVI_T0"], self.calc_ndvi(b08_tprev, b04_tprev))

        if b08_tprev is not None and b11_tprev is not None and "NDMI_T0" in results:
            results["dNDMI"] = self.calc_delta(results["NDMI_T0"], self.calc_ndmi(b08_tprev, b11_tprev))
            results["dMSI"] = self.calc_delta(results["MSI_T0"], self.calc_msi(b11_tprev, b08_tprev))

        if b08_tprev is not None and b12_tprev is not None and "NBR_T0" in results:
            results["dNBR"] = self.calc_delta(results["NBR_T0"], self.calc_nbr(b08_tprev, b12_tprev))

        if b08_tprev is not None and b11_tprev is not None and b12_tprev is not None and "NMDI_T0" in results:
            results["dNMDI"] = self.calc_delta(results["NMDI_T0"], self.calc_nmdi(b08_tprev, b11_tprev, b12_tprev))

        return results