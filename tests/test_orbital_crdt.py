import unittest

from orbital_crdt.analysis import false_accept_curve
from orbital_crdt.revocation_race import run_revocation_race
from orbital_crdt.scheduler import SatelliteVisibilityScheduler
from orbital_crdt.simulation import simulate_three_node_convergence


class OrbitalCRDTTests(unittest.TestCase):
    def test_scheduler_queues_until_next_window(self):
        scheduler = SatelliteVisibilityScheduler.periodic(first_window_start_min=47.0, orbital_period_min=94.6)
        self.assertAlmostEqual(scheduler.delivery_time(0.0), 47.0 * 60.0 + 0.6)

    def test_convergence_bounded_by_orbital_period(self):
        result = simulate_three_node_convergence(orbital_period_min=90.0, next_pass_offset_min=90.0)
        self.assertLessEqual(result["max_convergence_time_sec"], 90.0 * 60.0 + 1.0)
        self.assertTrue(result["all_converged"])

    def test_revocation_race_false_accept_then_reject(self):
        race = run_revocation_race()
        self.assertTrue(race["before_accept"])
        self.assertFalse(race["after_accept"])
        self.assertAlmostEqual(race["false_accept_window_min"], 47.01, places=2)

    def test_false_accept_curve_shape(self):
        rows = false_accept_curve(orbital_period_min=90.0, pass_duration_min=10.0, step_min=10.0)
        self.assertEqual(rows[0]["false_accept_probability"], 0.0)
        self.assertGreater(rows[-1]["false_accept_probability"], 0.9)


if __name__ == "__main__":
    unittest.main()

