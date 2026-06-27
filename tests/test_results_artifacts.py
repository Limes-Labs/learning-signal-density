import json
import unittest
from pathlib import Path


class CommittedResultArtifactTests(unittest.TestCase):
    def test_confirmation_sweep_uses_disjoint_seeds_and_records_sample_aware_result(self) -> None:
        discovery = json.loads(Path("results/sample_budget_sweep.json").read_text())
        confirmation = json.loads(Path("results/confirmation_budget_sweep.json").read_text())

        self.assertTrue(set(discovery["seeds"]).isdisjoint(confirmation["seeds"]))
        self.assertEqual(confirmation["confirmation_of"], "results/sample_budget_sweep.json")
        self.assertEqual(confirmation["claim_scope"]["paper_ready_claim"], False)
        self.assertEqual(confirmation["claim_scope"]["fresh_seed_confirmation"], True)

        thresholds = confirmation["thresholds"]
        sample_aware = thresholds["sample_aware_self_ranked_induction"]
        self_ranked = thresholds["self_ranked_induction"]
        self.assertGreater(sample_aware["best_signed_gain"], self_ranked["best_signed_gain"])
        self.assertEqual(sample_aware["first_material_count_reaching_target"], 48)

        budget_24 = confirmation["budgets"]["24"]
        self.assertGreater(
            budget_24["sample_aware_self_ranked_induction"]["accuracy_improvement_over_majority_mean"],
            budget_24["self_ranked_induction"]["accuracy_improvement_over_majority_mean"],
        )


if __name__ == "__main__":
    unittest.main()
