import numpy as np


class IndexCalculator:
    """calculates multi-spectral optical and sar polarimetric indices."""

    @staticmethod
    def calc_delta(t0_arr: np.ndarray, tprev_arr: np.ndarray) -> np.ndarray:
        """computes temporal difference between t0 and tprev rasters."""
        return (t0_arr - tprev_arr).astype(np.float32)

    @staticmethod
    def calc_msi(b11: np.ndarray, b08: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """calculates moisture stress index (swir1 / nir)."""
        return ((b11 + eps) / (b08 + eps)).astype(np.float32)

    @staticmethod
    def calc_nmdi(
        b08: np.ndarray,
        b11: np.ndarray,
        b12: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """calculates normalized multi-band drought index."""
        diff_swir = b11 - b12
        numerator = b08 - diff_swir
        denominator = b08 + diff_swir
        return (numerator / (denominator + eps)).astype(np.float32)

    def compute_all_indices(
        self,
        b02: np.ndarray,
        b03: np.ndarray,
        b04: np.ndarray,
        b05: np.ndarray,
        b06: np.ndarray,
        b07: np.ndarray,
        b08: np.ndarray,
        b8a: np.ndarray,
        b11: np.ndarray,
        b12: np.ndarray,
        scl: np.ndarray | None = None,
        sar_vv: np.ndarray | None = None,
        sar_vh: np.ndarray | None = None,
        b04_tprev: np.ndarray | None = None,
        b08_tprev: np.ndarray | None = None,
        b11_tprev: np.ndarray | None = None,
        b12_tprev: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """computes static t0 indices and temporal deltas."""
        results: dict[str, np.ndarray] = {}

        # 1. optical indices (t0)
        results["NDVI_T0"] = self.calc_ndvi(b08, b04)
        results["NDMI_T0"] = self.calc_ndmi(b08, b11)
        results["NBR_T0"] = self.calc_nbr(b08, b12)
        results["NBR2_T0"] = self.calc_nbr2(b11, b12)
        results["EVI_T0"] = self.calc_evi(b08, b04, b02)
        results["NDRE_T0"] = self.calc_ndre(b08, b05)
        results["MSI_T0"] = self.calc_msi(b11, b08)
        results["NMDI_T0"] = self.calc_nmdi(b08, b11, b12)

        # 2. sar polarimetric indices
        if sar_vv is not None and sar_vh is not None:
            results["SAR_RATIO"] = self.calc_sar_ratio(sar_vh, sar_vv)
            results["SAR_RVI"] = self.calc_sar_rvi(sar_vv, sar_vh)

        # 3. temporal spectral changes (t0 - tprev)
        if b08_tprev is not None and b04_tprev is not None:
            results["dNDVI"] = self.calc_delta(
                results["NDVI_T0"], self.calc_ndvi(b08_tprev, b04_tprev)
            )

        if b08_tprev is not None and b11_tprev is not None:
            results["dNDMI"] = self.calc_delta(
                results["NDMI_T0"], self.calc_ndmi(b08_tprev, b11_tprev)
            )
            # moisture stress delta
            results["dMSI"] = self.calc_delta(
                results["MSI_T0"], self.calc_msi(b11_tprev, b08_tprev)
            )

        if b08_tprev is not None and b12_tprev is not None:
            results["dNBR"] = self.calc_delta(
                results["NBR_T0"], self.calc_nbr(b08_tprev, b12_tprev)
            )

        if b08_tprev is not None and b11_tprev is not None and b12_tprev is not None:
            # normalized multi-band drought delta
            results["dNMDI"] = self.calc_delta(
                results["NMDI_T0"],
                self.calc_nmdi(b08_tprev, b11_tprev, b12_tprev),
            )

        return results