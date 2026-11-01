"""Learning-rate bands for fine-tuning and the validation-based choice."""
import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import run_experiments as rx
import run_lr_sweep as sweep


class TestLrBands(unittest.TestCase):

    def test_every_grid_size_falls_in_a_band(self):
        self.assertEqual(rx.ft_lr_band(10), "small")
        self.assertEqual(rx.ft_lr_band(100), "small")
        self.assertEqual(rx.ft_lr_band(250), "medium")
        self.assertEqual(rx.ft_lr_band(1000), "medium")
        self.assertEqual(rx.ft_lr_band(2500), "large")
        self.assertEqual(rx.ft_lr_band(20000), "large")

    def test_grid_uses_the_chosen_rate_per_band(self):
        chosen = {"deberta": {"small": 5e-5, "large": 2e-5}}
        with mock.patch.object(rx, "load_learning_rates", return_value=chosen):
            exps = rx.generate_finetuning_commands()
        lr = {(e["model"], e["dataset"], e["num_labelled"]): e for e in exps}
        self.assertEqual(lr[("deberta", "sst2", 50)]["lr"], 5e-5)
        self.assertIn("5e-05", lr[("deberta", "sst2", 50)]["cmd_args"])
        self.assertEqual(lr[("deberta", "sst2", 5000)]["lr"], 2e-5)
        self.assertEqual(lr[("deberta", "sst2", 500)]["lr"], rx.DEFAULT_FT_LR)  # band not chosen
        self.assertEqual(lr[("bert", "sst2", 50)]["lr"], rx.DEFAULT_FT_LR)

    def test_learning_rates_file_round_trip(self):
        path = os.path.join(tempfile.mkdtemp(), "lr.json")
        self.assertEqual(rx.load_learning_rates(path), {})
        with open(path, "w") as f:
            json.dump({"learning_rates": {"bert": {"small": 3e-5}}}, f)
        self.assertEqual(rx.load_learning_rates(path), {"bert": {"small": 3e-5}})

    def test_sweep_runs_one_repeat_in_its_own_results_root(self):
        exps = sweep.sweep_experiments(["roberta"], ["sst2"], [1e-5, 3e-5], "results_lr_sweep")
        self.assertEqual(len(exps), len(rx.FT_LR_BANDS) * 2)
        cmd = exps[0]["cmd_args"]
        self.assertEqual(cmd[cmd.index("--rskf_repeats") + 1], "1")
        self.assertEqual(cmd[cmd.index("--results_root") + 1], "results_lr_sweep")
        self.assertEqual(cmd[cmd.index("--configuration_name") + 1], "lr_1e-05_num_samples_100")
        self.assertEqual(cmd[cmd.index("--holdout_val") + 1], "1")
        grid = rx.finetuning_experiment("sst2", "roberta", 100, 1e-5)
        self.assertNotIn("--holdout_val", grid["cmd_args"])


def _flag(exp, flag):
    cmd = exp["cmd_args"]
    return cmd[cmd.index(flag) + 1]


class TestItLearningRate(unittest.TestCase):
    """The LoRA LR of instruction tuning is chosen per model and band, like fine-tuning's."""

    def test_it_grid_uses_the_chosen_rate_per_band(self):
        chosen = {"qwen_9b": {"small": 5e-4}}
        with mock.patch.object(rx, "load_learning_rates", return_value=chosen):
            exps = rx.generate_it_modern_commands()
        lr = {(e["model"], e["model_size"], e["dataset"], e["num_labelled"]): e for e in exps}
        self.assertEqual(lr[("qwen", "9b", "sst2", 50)]["lr"], 5e-4)
        self.assertEqual(float(_flag(lr[("qwen", "9b", "sst2", 50)], "--lr")), 5e-4)
        self.assertEqual(lr[("qwen", "9b", "sst2", 500)]["lr"], rx.DEFAULT_IT_LR)  # band not chosen
        self.assertEqual(lr[("phi", "mini", "sst2", 50)]["lr"], rx.DEFAULT_IT_LR)  # each model is its own key
        self.assertNotIn(("qwen", "4b", "sst2", 50), lr)  # dropped from the modern grid

    def test_without_a_sweep_lora_does_not_use_the_full_fine_tuning_rate(self):
        with mock.patch.object(rx, "load_learning_rates", return_value={}):
            exps = rx.generate_it_modern_commands()
        self.assertEqual({float(_flag(e, "--lr")) for e in exps}, {rx.DEFAULT_IT_LR})
        self.assertGreaterEqual(rx.DEFAULT_IT_LR, 1e-4)

    def test_it_sweep_scores_two_folds_on_validation_only(self):
        exps = sweep.sweep_experiments(["qwen_9b"], ["sst2"], [1e-4], "results_it_lr_sweep",
                                       approach="instruction_tuning")
        self.assertEqual(len(exps), len(rx.FT_LR_BANDS))
        small = exps[0]
        self.assertEqual((_flag(small, "--model"), _flag(small, "--model_size")), ("qwen", "9b"))
        self.assertEqual(_flag(small, "--configuration_name"), "lr_0.0001_num_samples_100")
        self.assertEqual(_flag(small, "--rskf_splits"), str(sweep.IT_SWEEP_SPLITS))
        self.assertEqual(small["rskf_splits"], sweep.IT_SWEEP_SPLITS)  # what is_already_completed counts
        self.assertEqual(_flag(small, "--holdout_val"), "1")
        self.assertEqual(_flag(small, "--it_val_only"), "1")
        self.assertEqual(_flag(small, "--results_root"), "results_it_lr_sweep")
        grid = rx.it_modern_experiment("sst2", "qwen", "9b", 100, 2e-4)
        for flag in ("--holdout_val", "--it_val_only"):
            self.assertNotIn(flag, grid["cmd_args"])

    def test_it_sweep_results_are_read_per_model_and_size(self):
        root = tempfile.mkdtemp()
        [exp] = [e for e in sweep.sweep_experiments(["qwen_9b"], ["sst2"], [1e-4], root,
                                                    approach="instruction_tuning")
                 if e["band"] == "small"]
        os.makedirs(rx.results_dir(exp))
        pd.DataFrame({"val_loss": [0.2, 0.4], "f1_prompting": [float("nan")] * 2}).to_parquet(
            os.path.join(rx.results_dir(exp), "summary.parquet"))
        [row] = sweep.load_sweep_results([exp]).to_dict("records")
        self.assertEqual((row["model"], row["band"], row["lr"]), ("qwen_9b", "small", 1e-4))
        self.assertAlmostEqual(row["val_loss"], 0.3)
        self.assertEqual(row["val_folds"], 2)


class TestSelect(unittest.TestCase):

    def test_lowest_mean_rank_of_validation_loss_wins(self):
        rows = [
            # 3e-5 wins on both datasets despite a different loss scale per dataset
            ("deberta", "small", "sst2", 1e-5, 0.60), ("deberta", "small", "sst2", 3e-5, 0.40),
            ("deberta", "small", "snips", 1e-5, 1.90), ("deberta", "small", "snips", 3e-5, 1.20),
            # test F1 must not drive the choice
            ("bert", "small", "sst2", 1e-5, 0.50), ("bert", "small", "sst2", 3e-5, 0.55),
        ]
        table = pd.DataFrame(rows, columns=["model", "band", "dataset", "lr", "val_loss"])
        table["test_f1_macro"] = [0.1, 0.2, 0.1, 0.2, 0.1, 0.9]
        chosen, _ = sweep.select(table)
        self.assertEqual(chosen["deberta"]["small"], 3e-5)
        self.assertEqual(chosen["bert"]["small"], 1e-5)


if __name__ == "__main__":
    unittest.main()
