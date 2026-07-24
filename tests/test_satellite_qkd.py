import unittest

from satellite_qkd.bb84_satellite import secure_key_rate_bps
from satellite_qkd.orbital_dynamics import (
    atmospheric_loss_dB,
    expected_qber,
    pass_parameters_from_config,
    slant_range,
    total_loss_timeseries,
)
from satellite_qkd.pass_simulator import simulate_pass


class SatelliteQKDTests(unittest.TestCase):
    def test_slant_range_matches_expected_scale(self):
        self.assertAlmostEqual(slant_range(90.0, 500.0), 500.0, delta=1.0)
        self.assertGreater(slant_range(20.0, 500.0), 1000.0)

    def test_atmospheric_loss_is_positive(self):
        self.assertGreater(atmospheric_loss_dB(45.0, 0.07, 8.0), 0.0)

    def test_expected_qber_acceptance_bands(self):
        self.assertGreaterEqual(expected_qber(90.0, 1e-17), 0.02)
        self.assertLessEqual(expected_qber(90.0, 1e-17), 0.04)
        self.assertGreater(expected_qber(20.0, 1e-17), 0.08)
        self.assertGreater(expected_qber(70.0, 2e-13), 0.11)

    def test_pass_timeseries_has_one_second_resolution(self):
        params = pass_parameters_from_config(pass_duration_sec=10)
        series = total_loss_timeseries(params)
        self.assertEqual(len(series), 10)
        self.assertIn("loss_dB", series[0])

    def test_secure_rate_aborts_above_threshold(self):
        self.assertEqual(secure_key_rate_bps(25.0, 0.12), 0.0)
        self.assertGreater(secure_key_rate_bps(22.0, 0.03), 0.0)

    def test_clear_pass_yield_acceptance_range(self):
        result = simulate_pass(
            {"pass_duration_sec": 600, "cn2": 1e-17, "max_elevation_deg": 90.0},
            seed=7,
        )
        self.assertGreaterEqual(result["summary"]["total_secure_key_bits"], 100_000)
        self.assertLessEqual(result["summary"]["total_secure_key_bits"], 300_000)


if __name__ == "__main__":
    unittest.main()
