import numpy as np
import pytest

from src.data_pipeline.index_calculator import IndexCalculator


class TestIndexCalculatorUnit:
    """Unit tests with synthetic edge-case arrays."""

    @pytest.fixture
    def calc(self):
        return IndexCalculator(
            epsilon=1e-7,
            reflectance_scale=10000.0,
            auto_detect_scale=False,
            sar_ratio_max=10.0,
        )

    @pytest.fixture
    def mock_grid(self):
        """Creates 64x64 synthetic satellite channels."""
        h, w = 64, 64
        return {
            "b02": np.full((h, w), 500.0, dtype=np.float32),
            "b04": np.full((h, w), 800.0, dtype=np.float32),
            "b05": np.full((h, w), 1200.0, dtype=np.float32),
            "b08": np.full((h, w), 2500.0, dtype=np.float32),
            "b8a": np.full((h, w), 2600.0, dtype=np.float32),
            "b11": np.full((h, w), 1500.0, dtype=np.float32),
            "b12": np.full((h, w), 900.0, dtype=np.float32),
            "scl": np.random.choice([0, 2, 3, 4, 5, 6, 8, 9, 10, 11], size=(h, w)).astype(np.uint8),
            "sar_vv": np.full((h, w), 0.08, dtype=np.float32),
            "sar_vh": np.full((h, w), 0.02, dtype=np.float32),
        }

    
    # 1. Optical & Moisture Tests
    

    def test_ndvi_range_and_correctness(self, calc, mock_grid):
        ndvi = calc.calc_ndvi(mock_grid["b08"], mock_grid["b04"])
        expected = (2500.0 - 800.0) / (2500.0 + 800.0)

        assert ndvi.dtype == np.float32
        assert np.allclose(ndvi, expected, atol=1e-5)
        assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)

    def test_ndre_narrow_band(self, calc, mock_grid):
        ndre = calc.calc_ndre(mock_grid["b8a"], mock_grid["b05"])
        expected = (2600.0 - 1200.0) / (2600.0 + 1200.0)

        assert ndre.dtype == np.float32
        assert np.allclose(ndre, expected, atol=1e-5)
        assert np.all(ndre >= -1.0) and np.all(ndre <= 1.0)

    def test_msi_clamping_and_bounds(self, calc):
        # High moisture stress: high SWIR, near-zero NIR
        b11 = np.array([[1000.0, 5000.0]], dtype=np.float32)
        b08 = np.array([[10.0, 0.0]], dtype=np.float32)

        msi = calc.calc_msi(b11, b08)
        assert msi.dtype == np.float32
        assert np.all(msi >= 0.0)
        assert np.all(msi <= 5.0)
        assert msi[0, 1] == 0.0  # Division by 0 handled safely

    def test_zero_division_and_nans(self, calc):
        zeros = np.zeros((10, 10), dtype=np.float32)
        nans = np.full((10, 10), np.nan, dtype=np.float32)

        ndvi_zeros = calc.calc_ndvi(zeros, zeros)
        assert not np.any(np.isnan(ndvi_zeros))
        assert np.all(ndvi_zeros == 0.0)

        ndvi_nans = calc.calc_ndvi(nans, nans)
        assert not np.any(np.isnan(ndvi_nans))
        assert np.all(ndvi_nans == 0.0)

    
    # 2. Boolean Masks Tests
    

    def test_boolean_masks_types_and_mapping(self, calc):
        scl = np.array([
            [11, 6],   # Snow, Water
            [2, 3],    # Dark area, Cloud shadow
            [8, 4],    # Cloud medium, Vegetation
        ], dtype=np.uint8)

        snow_mask = calc.calc_snow_mask(scl)
        invalid_mask = calc.calc_optical_invalid_mask(scl)
        water_mask = calc.calc_water_mask(scl)

        assert snow_mask.dtype == bool
        assert invalid_mask.dtype == bool
        assert water_mask.dtype == bool

        assert snow_mask[0, 0] is np.True_ and snow_mask[0, 1] is np.False_
        assert water_mask[0, 1] is np.True_ and water_mask[0, 0] is np.False_
        assert invalid_mask[1, 0] is np.True_  # SCL 2
        assert invalid_mask[1, 1] is np.True_  # SCL 3
        assert invalid_mask[2, 0] is np.True_  # SCL 8
        assert invalid_mask[2, 1] is np.False_ # SCL 4 (Vegetation is valid)

    
    # 3. Temporal Deltas Tests
    

    def test_temporal_delta_bounds(self, calc):
        t0 = np.array([[1.0, -1.0]], dtype=np.float32)
        tprev = np.array([[-1.0, 1.0]], dtype=np.float32)

        delta = calc.calc_delta(t0, tprev)
        assert delta.dtype == np.float32
        assert delta[0, 0] == 2.0
        assert delta[0, 1] == -2.0
        assert np.all(delta >= -2.0) and np.all(delta <= 2.0)

    
    # 4. SAR Polarimetric Tests
    

    def test_sar_rvi_physical_range(self, calc):
        vv = np.array([[0.1, 0.05, 0.0]], dtype=np.float32)
        vh = np.array([[0.05, 0.05, 0.02]], dtype=np.float32)

        rvi = calc.calc_sar_rvi(vh, vv)
        assert rvi.dtype == np.float32
        assert np.all(rvi >= 0.0)
        assert np.all(rvi <= 4.0)

        # RVI = 4 * 0.05 / (0.1 + 0.05) = 0.2 / 0.15 ≈ 1.3333 (> 1.0 preserve check)
        assert np.isclose(rvi[0, 0], 4.0 * 0.05 / 0.15, atol=1e-4)


class TestPipelineIntegration:
    """Integration test connecting SentinelFetcher to IndexCalculator."""

    def test_sentinel_fetcher_to_index_calculator_pipeline(self):
        from src.data_pipeline.sentinel_fetcher import SentinelFetcher

        fetcher = SentinelFetcher()
        calc = IndexCalculator()

        # Edmonton area test scene
        raw_scene = fetcher.fetch_all_radar_optical(
            lat=53.5461,
            lon=-113.4937,
            target_date="2023-06-15",
        )

        assert bool(raw_scene), "SentinelFetcher returned empty dictionary. Check network/planetary computer access."

        t0 = raw_scene["bands_t0"]
        tprev = raw_scene.get("bands_tprev", {})
        sar = raw_scene.get("sar_bands", {})

        results = calc.compute_all_indices(
            b02_t0=t0["B02"],
            b04_t0=t0["B04"],
            b08_t0=t0["B08"],
            b8a_t0=t0["B8A"],
            b11_t0=t0["B11"],
            b12_t0=t0["B12"],
            scl_t0=t0["SCL"],
            b05_t0=t0.get("B05"),
            b04_tprev=tprev.get("B04"),
            b08_tprev=tprev.get("B08"),
            b11_tprev=tprev.get("B11"),
            b12_tprev=tprev.get("B12"),
            sar_vv=sar.get("SAR_VV"),
            sar_vh=sar.get("SAR_VH"),
        )

        # Verify key expected outputs
        expected_keys = [
            "NDVI_T0", "NDRE_T0", "NDMI_T0", "MSI_T0", "NBR_T0", "NBR2_T0", "NMDI_T0", "EVI_T0",
            "MASK_SNOW", "MASK_OPTICAL_INVALID", "MASK_WATER",
            "dNDVI", "dNDMI", "dNBR",
            "SAR_RATIO", "SAR_RVI"
        ]

        for k in expected_keys:
            assert k in results, f"Missing expected key: {k}"
            arr = results[k]
            assert arr.shape == (2048, 2048), f"Incorrect shape {arr.shape} for {k}"
            assert not np.any(np.isinf(arr)), f"Found Infs in {k}"


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("1. RUNNING SYNTHETIC UNIT TESTS")
    print("=" * 70)

    unit = TestIndexCalculatorUnit()
    fixture_calc = unit.calc()
    grid = unit.mock_grid(fixture_calc)

    unit.test_ndvi_range_and_correctness(fixture_calc, grid)
    unit.test_ndre_narrow_band(fixture_calc, grid)
    unit.test_msi_clamping_and_bounds(fixture_calc)
    unit.test_zero_division_and_nans(fixture_calc)
    unit.test_boolean_masks_types_and_mapping(fixture_calc)
    unit.test_temporal_delta_bounds(fixture_calc)
    unit.test_sar_rvi_physical_range(fixture_calc)

    print(" -> All synthetic unit tests passed successfully!\n")

    print("=" * 70)
    print("2. RUNNING FULL INTEGRATION PIPELINE TEST")
    print("=" * 70)

    try:
        integration = TestPipelineIntegration()
        integration.test_sentinel_fetcher_to_index_calculator_pipeline()
        print(" -> Integration pipeline test with Planetary Computer passed successfully!")
    except Exception as e:
        print(f" -> Integration test failed with error: {e}")