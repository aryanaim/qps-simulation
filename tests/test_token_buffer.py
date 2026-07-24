import unittest

from token_buffer.buffer import Token, TokenBuffer
from token_buffer.simulation import cloudy_passes_survivable, simulate_buffer


class TokenBufferTests(unittest.TestCase):
    def test_fifo_consumes_oldest_token(self):
        buffer = TokenBuffer(capacity=10, staleness_max_hours=24)
        buffer.add_tokens([Token("a", 0.0, 1), Token("b", 1.0, 1)], 2.0)
        self.assertEqual(buffer.consume_token(3.0).token_id, "a")

    def test_stale_tokens_expire(self):
        buffer = TokenBuffer(capacity=10, staleness_max_hours=1)
        buffer.add_tokens([Token("old", 0.0, 1), Token("new", 3600.0, 2)], 3600.0)
        self.assertEqual(buffer.expire_stale(7201.0), 2)
        self.assertTrue(buffer.is_degraded())

    def test_simulation_emits_contract_rows(self):
        sim = simulate_buffer(days=0.1, consumption_rate_per_hour=10, pass_contract_A={"tokens_issued": 100}, seed=5)
        self.assertIn("timeseries", sim)
        self.assertIn("contract_B", sim)

    def test_resilience_acceptance_shape(self):
        self.assertGreaterEqual(cloudy_passes_survivable(50, 502), 3)
        self.assertGreaterEqual(cloudy_passes_survivable(100, 502), 1)
        self.assertLessEqual(cloudy_passes_survivable(100, 502), 2)


if __name__ == "__main__":
    unittest.main()
