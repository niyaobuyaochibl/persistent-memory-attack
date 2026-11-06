from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stickyrec.analysis import analyze_user_metrics, paired_bootstrap
from stickyrec.attack_analysis import attack_lift_table
from stickyrec.llm_analysis import analyze_llm_user_metrics


class AnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_constant_difference(self) -> None:
        low, high, p_value = paired_bootstrap(np.array([0.2, 0.2]), seed=1)
        self.assertEqual((low, high, p_value), (0.2, 0.2, 0.0))

    def test_user_metrics_are_paired_by_seed_and_user(self) -> None:
        rows = []
        values = {"raw_memory": 0.5, "eam": 0.1, "eam_delayed": 0.0, "no_memory": 0.0}
        for policy, path in values.items():
            rows.append(
                {
                    "drift_mode": "abrupt",
                    "feedback_model": "deterministic",
                    "seed": 1,
                    "user_id": 7,
                    "policy": policy,
                    "path_dependence": path,
                    "head_cumulative_utility": 1.0,
                    "tail_cumulative_utility": 1.0,
                    "head_post_drift_mean_regret": 0.1,
                    "tail_post_drift_mean_regret": 0.1,
                }
            )
        effects = analyze_user_metrics(pd.DataFrame(rows))
        effect = effects[effects["comparison"] == "raw_minus_eam_path"].iloc[0]
        self.assertEqual(effect["pairs"], 1)
        self.assertAlmostEqual(effect["mean_difference"], 0.4)

    def test_llm_metrics_are_paired_by_seed_and_user(self) -> None:
        rows = []
        for seed in (1, 2):
            for user_id in (7, 8):
                for mode, path in {
                    "none": 0.0,
                    "raw": 0.5,
                    "eam": 0.2,
                    "summary": 0.3,
                }.items():
                    rows.append(
                        {
                            "model": "test-model",
                            "dataset": "ml-1m",
                            "drift_mode": "stable",
                            "feedback_model": "deterministic",
                            "seed": seed,
                            "user_id": user_id,
                            "memory_mode": mode,
                            "path_dependence": path,
                            "head_cumulative_utility": 1.0,
                            "tail_cumulative_utility": 1.0,
                        }
                    )
        effects = analyze_llm_user_metrics(pd.DataFrame(rows))
        effect = effects[effects["comparison"] == "raw_minus_eam_path"].iloc[0]
        self.assertEqual(effect["pairs"], 4)
        self.assertEqual(effect["seeds"], 2)
        self.assertAlmostEqual(effect["mean_difference"], 0.3)


    def test_attack_lift_table_reports_lift_and_defense(self) -> None:
        rows = []
        for seed in (1, 2):
            for user_id in range(6):
                # Raw memory is manipulable: attack arm share > clean arm.
                # ERSM neutralizes it: attack arm == clean arm.
                shares = {
                    ("raw", "head"): 0.6,
                    ("raw", "tail"): 0.3,
                    ("ersm", "head"): 0.3,
                    ("ersm", "tail"): 0.3,
                }
                for (mode, condition), share in shares.items():
                    rows.append(
                        {
                            "seed": seed,
                            "user_id": user_id,
                            "memory_mode": mode,
                            "condition": condition,
                            "target_share": share,
                        }
                    )
        table = attack_lift_table(pd.DataFrame(rows))
        raw_lift = table[table["comparison"] == "attack_lift[raw]"].iloc[0]
        ersm_lift = table[table["comparison"] == "attack_lift[ersm]"].iloc[0]
        defense = table[table["comparison"] == "defense[raw-ersm]"].iloc[0]
        self.assertAlmostEqual(raw_lift["mean"], 0.3)
        self.assertAlmostEqual(ersm_lift["mean"], 0.0)
        self.assertAlmostEqual(defense["mean"], 0.3)
        self.assertEqual(int(raw_lift["pairs"]), 12)


if __name__ == "__main__":
    unittest.main()
