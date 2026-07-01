import io
import tarfile
import unittest

from learning_signal_density.newsgroups_experiment import (
    NewsRecord,
    parse_twenty_newsgroups_tar_gz,
    run_newsgroups_condition,
    run_newsgroups_seedset,
)


def _tiny_newsgroups_payload() -> bytes:
    docs = {
        "mini_newsgroups/sci.space/1": "Subject: Launch\nFrom: a@example.com\n\nOrbit rocket mission shuttle\n> quoted leak\nNASA orbit",
        "mini_newsgroups/sci.space/2": "Subject: Mars\n\nMars rover orbit space mission",
        "mini_newsgroups/sci.space/3": "Subject: Telescope\n\nTelescope galaxy orbit star",
        "mini_newsgroups/rec.autos/1": "Subject: Engine\n\nEngine wheel torque car",
        "mini_newsgroups/rec.autos/2": "Subject: Tire\n\nCar tire road engine",
        "mini_newsgroups/rec.autos/3": "Subject: Brakes\n\nBrake engine wheel road",
        "mini_newsgroups/talk.politics/1": "Subject: Vote\n\nElection policy senate vote",
        "mini_newsgroups/talk.politics/2": "Subject: Law\n\nPolicy court election law",
        "mini_newsgroups/talk.politics/3": "Subject: Debate\n\nDebate vote policy party",
    }
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, text in docs.items():
            encoded = text.encode("latin-1")
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
    return payload.getvalue()


class NewsgroupsExperimentTests(unittest.TestCase):
    def test_parse_twenty_newsgroups_tar_gz_strips_headers_and_quotes(self) -> None:
        records = parse_twenty_newsgroups_tar_gz(_tiny_newsgroups_payload(), corpus_root="mini_newsgroups")

        self.assertEqual(len(records), 9)
        self.assertEqual(sorted({record.label for record in records}), ["rec.autos", "sci.space", "talk.politics"])
        first = next(record for record in records if record.record_id == "sci.space/1")
        self.assertNotIn("Subject:", first.text)
        self.assertNotIn("From:", first.text)
        self.assertNotIn("quoted leak", first.text)
        self.assertIn("Orbit rocket mission", first.text)

    def test_run_newsgroups_condition_records_curriculum_cost_without_heldout_selection(self) -> None:
        records = parse_twenty_newsgroups_tar_gz(_tiny_newsgroups_payload(), corpus_root="mini_newsgroups")

        row = run_newsgroups_condition(
            records=records,
            seed=17,
            condition="length_curriculum_sample",
            train_budget=3,
            validation_per_class=1,
            heldout_per_class=1,
            epochs=2,
        )

        self.assertEqual(row["dataset_name"], "Twenty Newsgroups")
        self.assertEqual(row["condition"], "length_curriculum_sample")
        self.assertEqual(row["external_events"], 3)
        self.assertEqual(row["heldout_used_for_selection"], False)
        self.assertGreater(row["selection_cost_tokens"], 0)
        self.assertGreater(row["charged_compute_units"], row["internal_tokens"])

    def test_seedset_artifact_compares_random_curriculum_retrieval_and_selector(self) -> None:
        records = parse_twenty_newsgroups_tar_gz(_tiny_newsgroups_payload(), corpus_root="mini_newsgroups")

        artifact = run_newsgroups_seedset(
            records=records,
            seeds=(17, 19),
            train_budgets=(3,),
            conditions=(
                "random_sample",
                "class_balanced_sample",
                "length_curriculum_sample",
                "prototype_retrieval_sample",
                "validation_selector",
            ),
            validation_per_class=1,
            heldout_per_class=1,
            epochs=2,
            proxy_epochs=1,
        )

        self.assertEqual(artifact["dataset"]["name"], "Twenty Newsgroups")
        self.assertEqual(artifact["claim_scope"]["real_dataset"], True)
        self.assertEqual(artifact["claim_scope"]["heldout_used_for_selection"], False)
        budget = artifact["budgets"]["3"]["conditions"]
        self.assertEqual(set(budget), set(artifact["condition_scope"]))
        self.assertGreater(
            budget["validation_selector"]["validation_tuning_cost_tokens_mean"],
            budget["random_sample"]["validation_tuning_cost_tokens_mean"],
        )
        self.assertIn("prototype_retrieval_sample", artifact["condition_scope"])


if __name__ == "__main__":
    unittest.main()
