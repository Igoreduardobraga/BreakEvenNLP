import unittest
import os
import shutil
import tempfile
import csv
from pathlib import Path

class TestResultStore(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.exp_meta = {
            "experiment_name": "test_exp",
            "experiment_type": "prompting",
            "model_name": "llama3_8b",
            "dataset": "agnews",
            "factor": "sample_order",
            "configuration_name": "sample_size_100",
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_init_creates_directories_and_metadata(self):
        """Verifies that ResultStore initializes with correct path and metadata."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            **self.exp_meta
        )

        self.assertEqual(store.experiment_name, "test_exp")
        self.assertEqual(store.experiment_type, "prompting")
        self.assertEqual(store.model_name, "llama3_8b")
        self.assertEqual(store.dataset, "agnews")
        self.assertEqual(store.factor, "sample_order")
        self.assertTrue(os.path.exists(self.test_dir))

    def test_record_fold_creates_summary_table_atomically(self):
        """Verifies that record_fold writes the expected columns to summary table."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )

        factor_seeds = {
            "data_split_seed": 101,
            "label_choice_seed": 102,
            "sample_choice_seed": 103,
            "sample_order_seed": 104,
            "model_initialisation_seed": 105,
            "model_randomness_seed": 106,
        }
        metrics = {"f1_macro": 0.855}

        store.record_fold(
            repeat=0,
            fold=1,
            run_seed_used=42,
            randomness_factor_seeds=factor_seeds,
            metrics=metrics,
            real=[0, 1, 0, 1],
            predicted=[0, 1, 0, 0],
            duration_seconds=12.34,
        )

        summary_csv = os.path.join(self.test_dir, "summary.csv")
        self.assertTrue(os.path.exists(summary_csv), "summary.csv should exist")

        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["experiment_name"], "test_exp")
        self.assertEqual(row["experiment_type"], "prompting")
        self.assertEqual(row["model"], "llama3_8b")
        self.assertEqual(row["dataset"], "agnews")
        self.assertEqual(row["factor"], "sample_order")
        self.assertEqual(row["rskf_repeat"], "0")
        self.assertEqual(row["rskf_fold"], "1")
        self.assertEqual(row["run_seed_used"], "42")
        self.assertEqual(row["sample_order_seed"], "104")
        self.assertAlmostEqual(float(row["f1_macro"]), 0.855, places=3)
        self.assertAlmostEqual(float(row["duration_seconds"]), 12.34, places=2)
        self.assertIn("timestamp", row)
        self.assertIn("git_commit", row)

    def test_record_multiple_folds_appends_incrementally(self):
        """Verifies that multiple folds are appended without overwriting previous data."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}

        # Record fold 0
        store.record_fold(
            repeat=0, fold=0, run_seed_used=1,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.80},
            real=[0, 1], predicted=[0, 1],
            duration_seconds=5.0
        )

        # Record fold 1
        store.record_fold(
            repeat=0, fold=1, run_seed_used=2,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.90},
            real=[0, 1], predicted=[1, 1],
            duration_seconds=6.0
        )

        summary_csv = os.path.join(self.test_dir, "summary.csv")
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rskf_fold"], "0")
        self.assertAlmostEqual(float(rows[0]["f1_macro"]), 0.80)
        self.assertEqual(rows[1]["rskf_fold"], "1")
        self.assertAlmostEqual(float(rows[1]["f1_macro"]), 0.90)

    def test_instruction_tuning_metrics_recorded(self):
        """Verifies that f1_prompting and f1_macro_icl are recorded properly."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **{**self.exp_meta, "experiment_type": "instruction_tuning"}
        )

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}

        store.record_fold(
            repeat=0, fold=0, run_seed_used=1,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_prompting": 0.78, "f1_macro_icl": 0.86},
            real=[0, 1], predicted=[0, 1],
            duration_seconds=10.0
        )

        summary_csv = os.path.join(self.test_dir, "summary.csv")
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["f1_prompting"]), 0.78)
        self.assertAlmostEqual(float(rows[0]["f1_macro_icl"]), 0.86)

    def test_atomic_write_leaves_no_stray_tmp_files(self):
        """Verifies that staging files are cleaned up after atomic commit."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}

        store.record_fold(
            repeat=0, fold=0, run_seed_used=1,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.80},
            real=[0, 1], predicted=[0, 1]
        )

        files = os.listdir(self.test_dir)
        tmp_files = [f for f in files if f.endswith(".tmp") or ".tmp" in f]
        self.assertEqual(len(tmp_files), 0, "Staging temp files should be cleanly renamed/removed")

    def test_fallback_to_csv_when_parquet_engine_missing(self):
        """Verifies that storage_format='parquet' falls back to csv if pyarrow is unavailable."""
        from unittest.mock import patch
        import sys
        from result_store import ResultStore

        with patch.dict(sys.modules, {"pyarrow": None}):
            store = ResultStore(
                results_path=self.test_dir,
                storage_format="parquet",
                **self.exp_meta
            )
            self.assertEqual(store.summary_file.name, "summary.csv")

    def test_parquet_atomic_write_with_mock(self):
        """Verifies parquet branch execution when pyarrow and pandas are available."""
        from unittest.mock import MagicMock, patch
        import sys
        from result_store import ResultStore

        mock_pa = MagicMock()
        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_pd.DataFrame.return_value = mock_df

        with patch.dict(sys.modules, {"pyarrow": mock_pa, "pandas": mock_pd}):
            store = ResultStore(
                results_path=self.test_dir,
                storage_format="parquet",
                **self.exp_meta
            )
            self.assertEqual(store.summary_file.name, "summary.parquet")

            factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
            store.record_fold(
                repeat=0, fold=0, run_seed_used=1,
                randomness_factor_seeds=factor_seeds,
                metrics={"f1_macro": 0.85},
                real=[0, 1], predicted=[0, 1]
            )

            self.assertEqual(mock_pd.DataFrame.call_count, 2)
            self.assertEqual(mock_df.to_parquet.call_count, 2)
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "summary.parquet")))
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "predictions.parquet")))

    def test_legacy_json_written_by_default(self):
        """Verifies that repeat_{r}_fold_{k}/results.json matches the legacy structure."""
        import json
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            legacy_json=True,
            **self.exp_meta
        )

        factor_seeds = {
            "data_split_seed": 10,
            "label_choice_seed": 11,
            "sample_choice_seed": 12,
            "sample_order_seed": 13,
            "model_initialisation_seed": 14,
            "model_randomness_seed": 15,
        }

        store.record_fold(
            repeat=0,
            fold=2,
            run_seed_used=99,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.88},
            real=[0, 1, 1],
            predicted=[0, 1, 0],
            decodeds=["class_a", "class_b", "class_a"],
        )

        legacy_path = os.path.join(self.test_dir, "repeat_0_fold_2", "results.json")
        self.assertTrue(os.path.exists(legacy_path), "Legacy results.json must exist")

        with open(legacy_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["f1_macro"], 0.88)
        self.assertEqual(data["real"], [0, 1, 1])
        self.assertEqual(data["predicted"], [0, 1, 0])
        self.assertEqual(data["base_model"], "llama3_8b")
        self.assertEqual(data["rskf_repeat"], 0)
        self.assertEqual(data["rskf_fold"], 2)
        self.assertEqual(data["run_seed_used"], 99)
        self.assertEqual(data["decodeds"], ["class_a", "class_b", "class_a"])
        self.assertEqual(data["sample_order_seed"], 13)

    def test_legacy_json_omitted_when_disabled(self):
        """Verifies that legacy folder and JSON are not written when legacy_json=False."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            legacy_json=False,
            **self.exp_meta
        )

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
        store.record_fold(
            repeat=1,
            fold=0,
            run_seed_used=55,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.82},
            real=[0, 1],
            predicted=[0, 1],
        )

        legacy_path = os.path.join(self.test_dir, "repeat_1_fold_0")
        self.assertFalse(os.path.exists(legacy_path), "Legacy folder should not be created when legacy_json=False")

    def test_has_fold_lifecycle(self):
        """Verifies has_fold returns False before recording and True after recording."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )

        self.assertFalse(store.has_fold(repeat=0, fold=3))

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
        store.record_fold(
            repeat=0,
            fold=3,
            run_seed_used=77,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.75},
            real=[1, 1],
            predicted=[1, 1],
        )

        self.assertTrue(store.has_fold(repeat=0, fold=3))

    def test_load_fold_details(self):
        """Verifies load_fold_details retrieves raw targets, predictions and decodeds."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
        store.record_fold(
            repeat=0,
            fold=4,
            run_seed_used=88,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.91},
            real=[1, 0, 1],
            predicted=[1, 1, 1],
            decodeds=["yes", "no", "yes"],
        )

        details = store.load_fold_details(repeat=0, fold=4)
        self.assertEqual(details["real"], [1, 0, 1])
        self.assertEqual(details["predicted"], [1, 1, 1])
        self.assertEqual(details["decodeds"], ["yes", "no", "yes"])

    def test_load_summary_records_list(self):
        """Verifies load_summary returns parsed dicts when as_dataframe=False."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
        store.record_fold(
            repeat=0, fold=0, run_seed_used=10,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.81},
            real=[0, 1], predicted=[0, 1],
            duration_seconds=5.2,
        )
        store.record_fold(
            repeat=0, fold=1, run_seed_used=20,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.89},
            real=[0, 1], predicted=[1, 1],
            duration_seconds=6.1,
        )

        summary = store.load_summary(as_dataframe=False)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["rskf_fold"], 0)
        self.assertEqual(summary[1]["rskf_fold"], 1)
        self.assertAlmostEqual(summary[0]["f1_macro"], 0.81)
        self.assertAlmostEqual(summary[1]["f1_macro"], 0.89)
        self.assertAlmostEqual(summary[0]["duration_seconds"], 5.2)

    def test_load_summary_filtering(self):
        """Verifies load_summary applies filter conditions properly."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )

        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
        store.record_fold(
            repeat=0, fold=0, run_seed_used=10,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.81},
            real=[0, 1], predicted=[0, 1],
        )
        store.record_fold(
            repeat=1, fold=0, run_seed_used=20,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.89},
            real=[0, 1], predicted=[1, 1],
        )

        # Filter by repeat
        filtered = store.load_summary(filters={"rskf_repeat": 1}, as_dataframe=False)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["rskf_repeat"], 1)

    def test_load_summary_with_mocked_dataframe(self):
        """Verifies that load_summary invokes pd.read_csv when pandas is available."""
        from unittest.mock import MagicMock, patch
        import sys
        from result_store import ResultStore

        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_pd.read_csv.return_value = mock_df

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )
        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
        store.record_fold(
            repeat=0, fold=0, run_seed_used=10,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.81},
            real=[0, 1], predicted=[0, 1],
        )

        with patch.dict(sys.modules, {"pandas": mock_pd}):
            df = store.load_summary(as_dataframe=True)
            mock_pd.read_csv.assert_called_once()
            self.assertEqual(df, mock_df)

    def test_provenance_metadata_logged(self):
        """Verifies that git_commit and timestamp are populated."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )
        factor_seeds = {f"{k}_seed": 100 for k in ["data_split", "label_choice", "sample_choice", "sample_order", "model_initialisation", "model_randomness"]}
        store.record_fold(
            repeat=0, fold=0, run_seed_used=10,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.81},
            real=[0, 1], predicted=[0, 1],
            duration_seconds=3.14,
        )

        summary = store.load_summary(as_dataframe=False)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertTrue(len(row["git_commit"]) > 0)
        self.assertTrue(len(row["timestamp"]) > 0)
        self.assertAlmostEqual(row["duration_seconds"], 3.14)

    def test_record_fold_creates_predictions_file(self):
        """Verifies that record_fold writes sample-level qualitative data to predictions file."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )
        factor_seeds = {"data_split": 100, "label_choice": 100}
        store.record_fold(
            repeat=0, fold=0, run_seed_used=10,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.5},
            real=[1, 0, 1],
            predicted=[1, 1, -1],
            decodeds=["positive", "positive", "not sure"],
            inputs=["great film", "awful acting", "maybe ok"],
            prompts=["prompt 1", "prompt 2", "prompt 3"],
        )

        preds = store.load_predictions(as_dataframe=False)
        self.assertEqual(len(preds), 3)

        # Sample 0: correct
        self.assertEqual(preds[0]["input_text"], "great film")
        self.assertEqual(preds[0]["prompt"], "prompt 1")
        self.assertEqual(preds[0]["model_response"], "positive")
        self.assertEqual(int(preds[0]["predicted_label"]), 1)
        self.assertEqual(int(preds[0]["golden_label"]), 1)
        self.assertEqual(str(preds[0]["is_correct"]).lower(), "true")
        self.assertEqual(preds[0]["error_type"], "correct")

        # Sample 1: misclassification
        self.assertEqual(int(preds[1]["predicted_label"]), 1)
        self.assertEqual(int(preds[1]["golden_label"]), 0)
        self.assertEqual(str(preds[1]["is_correct"]).lower(), "false")
        self.assertEqual(preds[1]["error_type"], "misclassification")

        # Sample 2: parsing failure (-1)
        self.assertEqual(int(preds[2]["predicted_label"]), -1)
        self.assertEqual(str(preds[2]["is_correct"]).lower(), "false")
        self.assertEqual(preds[2]["error_type"], "parsing_failure")

    def test_load_predictions_filters(self):
        """Verifies filtering in load_predictions."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )
        store.record_fold(
            repeat=0, fold=0, run_seed_used=10,
            randomness_factor_seeds={},
            metrics={"f1_macro": 0.5},
            real=[1, 0],
            predicted=[1, -1],
            decodeds=["pos", "err"],
            inputs=["text 1", "text 2"],
            prompts=["p1", "p2"],
        )

        failures = store.load_predictions(filters={"error_type": "parsing_failure"}, as_dataframe=False)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["input_text"], "text 2")

    def test_factor_seeds_populated_with_or_without_suffix(self):
        """Verifies that factor seeds dictionary without '_seed' suffix is correctly captured."""
        from result_store import ResultStore

        store = ResultStore(
            results_path=self.test_dir,
            storage_format="csv",
            **self.exp_meta
        )
        # RNGController returns keys without '_seed'
        factor_seeds = {
            "data_split": 555,
            "label_choice": 666,
            "sample_choice": 777,
            "sample_order": 888,
            "model_initialisation": 999,
            "model_randomness": 111,
        }
        store.record_fold(
            repeat=0, fold=0, run_seed_used=10,
            randomness_factor_seeds=factor_seeds,
            metrics={"f1_macro": 0.9},
            real=[1], predicted=[1],
        )

        summary = store.load_summary(as_dataframe=False)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["data_split_seed"], 555)
        self.assertEqual(row["label_choice_seed"], 666)
        self.assertEqual(row["model_randomness_seed"], 111)


if __name__ == "__main__":
    unittest.main()
