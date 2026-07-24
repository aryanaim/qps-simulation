import unittest

from adversary.timing_analysis import build_contract_c_transcripts, estimate_linkability_advantage


class AdversaryTests(unittest.TestCase):
    def test_linkability_advantage_below_threshold_for_randomized_buffer(self):
        events = [
            {
                "timestamp": float(i * 60),
                "token_consumed_id": f"tok-{i}",
                "pass_source_id": i % 5,
            }
            for i in range(1000)
        ]
        transcripts = build_contract_c_transcripts(events, seed=1)
        rows = estimate_linkability_advantage(transcripts, consumption_policy="random")
        self.assertLessEqual(max(row["within_pass_advantage"] for row in rows), 0.55)
        self.assertLessEqual(max(row["cross_pass_advantage"] for row in rows), 0.52)


if __name__ == "__main__":
    unittest.main()

