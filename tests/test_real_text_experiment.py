import io
import unittest
import zipfile

from learning_signal_density.real_text_experiment import (
    REAL_TEXT_CONDITION_SCOPE,
    TextRecord,
    parse_sms_spam_collection_zip,
    run_real_text_condition,
    run_real_text_seedset,
    stratified_split,
)


def make_sms_zip(lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SMSSpamCollection", "\n".join(lines) + "\n")
        archive.writestr("readme", "fixture")
    return buffer.getvalue()


def make_easy_sms_records(ham_count: int = 90, spam_count: int = 36) -> tuple[TextRecord, ...]:
    records: list[TextRecord] = []
    for index in range(ham_count):
        records.append(
            TextRecord(
                record_id=f"ham-{index:03d}",
                label=False,
                text=f"see you at dinner project meeting coffee family note {index % 7}",
            )
        )
    for index in range(spam_count):
        records.append(
            TextRecord(
                record_id=f"spam-{index:03d}",
                label=True,
                text=f"urgent prize claim free cash voucher call now winner {index % 5}",
            )
        )
    return tuple(records)


class RealTextExperimentTests(unittest.TestCase):
    def test_parse_sms_spam_zip_preserves_labels_text_and_stable_ids(self) -> None:
        payload = make_sms_zip(
            [
                "ham\tGo until jurong point, crazy.. Available only in bugis.",
                "spam\tFree entry in 2 a wkly comp to win FA Cup final tkts.",
                "ham\tOk lar... Joking wif u oni...",
            ]
        )

        records = parse_sms_spam_collection_zip(payload)

        self.assertEqual([record.record_id for record in records], ["sms-000001", "sms-000002", "sms-000003"])
        self.assertEqual([record.label for record in records], [False, True, False])
        self.assertIn("FA Cup", records[1].text)

    def test_stratified_split_keeps_disjoint_train_validation_and_heldout_ids(self) -> None:
        records = make_easy_sms_records()

        split = stratified_split(records, seed=17, validation_size=20, heldout_size=24)

        train_ids = {record.record_id for record in split.train_pool}
        validation_ids = {record.record_id for record in split.validation}
        heldout_ids = {record.record_id for record in split.heldout}
        self.assertFalse(train_ids & validation_ids)
        self.assertFalse(train_ids & heldout_ids)
        self.assertFalse(validation_ids & heldout_ids)
        self.assertEqual(len(train_ids | validation_ids | heldout_ids), len(records))
        self.assertGreater(sum(record.label for record in split.validation), 0)
        self.assertGreater(sum(record.label for record in split.heldout), 0)

    def test_real_text_conditions_charge_selection_and_never_select_on_heldout(self) -> None:
        records = make_easy_sms_records()

        random_row = run_real_text_condition(
            records,
            seed=23,
            condition="random_sample",
            train_budget=24,
            validation_size=20,
            heldout_size=24,
            epochs=3,
        )
        balanced_row = run_real_text_condition(
            records,
            seed=23,
            condition="class_balanced_sample",
            train_budget=24,
            validation_size=20,
            heldout_size=24,
            epochs=3,
        )
        label_index_row = run_real_text_condition(
            records,
            seed=23,
            condition="label_index_balanced_sample",
            train_budget=24,
            validation_size=20,
            heldout_size=24,
            epochs=3,
        )
        selector_row = run_real_text_condition(
            records,
            seed=23,
            condition="validation_sample_selector",
            train_budget=24,
            validation_size=20,
            heldout_size=24,
            epochs=3,
        )
        label_index_selector_row = run_real_text_condition(
            records,
            seed=23,
            condition="validation_label_index_selector",
            train_budget=24,
            validation_size=20,
            heldout_size=24,
            epochs=3,
        )

        self.assertEqual(random_row["dataset_name"], "UCI SMS Spam Collection")
        self.assertEqual(random_row["condition"], "random_sample")
        self.assertEqual(balanced_row["train_spam_count"], 12)
        self.assertEqual(balanced_row["train_ham_count"], 12)
        self.assertEqual(balanced_row["majority_baseline_spam_f1"], 0.0)
        self.assertEqual(balanced_row["selection_cost_tokens"], balanced_row["train_pool_tokens"])
        self.assertEqual(label_index_row["train_spam_count"], 12)
        self.assertEqual(label_index_row["train_ham_count"], 12)
        self.assertEqual(label_index_row["selection_cost_tokens"], label_index_row["train_pool_size"])
        self.assertLess(label_index_row["selection_cost_tokens"], balanced_row["selection_cost_tokens"])
        self.assertGreater(selector_row["validation_tuning_cost_tokens"], 0)
        self.assertIn(selector_row["selected_sampling_policy"], {"random_sample", "class_balanced_sample"})
        self.assertGreater(label_index_selector_row["validation_tuning_cost_tokens"], 0)
        self.assertLess(
            label_index_selector_row["validation_tuning_cost_tokens"],
            selector_row["validation_tuning_cost_tokens"],
        )
        self.assertIn(
            label_index_selector_row["selected_sampling_policy"],
            {"random_sample", "label_index_balanced_sample"},
        )
        self.assertFalse(selector_row["heldout_used_for_selection"])
        self.assertGreaterEqual(selector_row["heldout_spam_f1"], 0.0)
        self.assertLessEqual(selector_row["heldout_spam_f1"], 1.0)
        self.assertIn("validation_sample_selector", REAL_TEXT_CONDITION_SCOPE)
        self.assertTrue(REAL_TEXT_CONDITION_SCOPE["validation_sample_selector"]["validation_used_for_policy_selection"])
        self.assertTrue(
            REAL_TEXT_CONDITION_SCOPE["validation_label_index_selector"]["validation_used_for_policy_selection"]
        )

    def test_seedset_artifact_marks_real_dataset_and_preserves_signed_metrics(self) -> None:
        records = make_easy_sms_records()

        result = run_real_text_seedset(
            records=records,
            seeds=(101, 103),
            train_budgets=(16, 24),
            validation_size=20,
            heldout_size=24,
            epochs=3,
        )

        self.assertEqual(result["title"], "Real Text Selection-Cost Pilot")
        self.assertEqual(result["dataset"]["name"], "UCI SMS Spam Collection")
        self.assertEqual(result["claim_scope"]["synthetic_domain"], False)
        self.assertEqual(result["claim_scope"]["real_dataset"], True)
        self.assertEqual(result["claim_scope"]["heldout_used_for_selection"], False)
        self.assertEqual(result["claim_scope"]["paper_ready_claim"], False)
        self.assertIn("24", result["budgets"])
        self.assertIn("label_index_balanced_sample", result["budgets"]["24"]["conditions"])
        self.assertIn("validation_sample_selector", result["budgets"]["24"]["conditions"])
        self.assertIn("validation_label_index_selector", result["budgets"]["24"]["conditions"])
        self.assertIn(
            "signed_learning_signal_density_per_1m_event_compute_mean",
            result["budgets"]["24"]["conditions"]["random_sample"],
        )


if __name__ == "__main__":
    unittest.main()
