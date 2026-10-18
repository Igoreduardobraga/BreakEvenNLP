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
        with mock.patch.object(rx, "load_ft_learning_rates", return_value=chosen):
            exps = rx.generate_finetuning_commands()
        lr = {(e["model"], e["dataset"], e["num_labelled"]): e for e in exps}
        self.assertEqual(lr[("deberta", "sst2", 50)]["lr"], 5e-5)
        self.assertIn("5e-05", lr[("deberta", "sst2", 50)]["cmd_args"])
        self.assertEqual(lr[("deberta", "sst2", 5000)]["lr"], 2e-5)
        self.assertEqual(lr[("deberta", "sst2", 500)]["lr"], rx.DEFAULT_FT_LR)  # band not chosen
        self.assertEqual(lr[("bert", "sst2", 50)]["lr"], rx.DEFAULT_FT_LR)

    def test_learning_rates_file_round_trip(self):
        path = os.path.join(tempfile.mkdtemp(), "lr.json")
        self.assertEqual(rx.load_ft_learning_rates(path), {})
        with open(path, "w") as f:
            json.dump({"learning_rates": {"bert": {"small": 3e-5}}}, f)
        self.assertEqual(rx.load_ft_learning_rates(path), {"bert": {"small": 3e-5}})

    def test_sweep_runs_one_repeat_in_its_own_results_root(self):
        exps = sweep.sweep_experiments(["roberta"], ["sst2"], [1e-5, 3e-5], "results_lr_sweep")
        self.assertEqual(len(exps), len(rx.FT_LR_BANDS) * 2)
        cmd = exps[0]["cmd_args"]
        self.assertEqual(cmd[cmd.index("--rskf_repeats") + 1], "1")
        self.assertEqual(cmd[cmd.index("--results_root") + 1], "results_lr_sweep")
        self.assertEqual(cmd[cmd.index("--configuration_name") + 1], "lr_1e-05_num_samples_100")


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
