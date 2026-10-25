"""Instruction tuning gives two curves in the break-even plot."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis import plot_break_even as pbe


def _summary(root, exp_type, exp_dir, model, **metrics):
    path = Path(root) / exp_type / exp_dir / "num_samples_10" / "toy" / "golden_model"
    path.mkdir(parents=True)
    row = {"experiment_type": exp_type, "model": model, "dataset": "toy",
           "configuration": "num_samples_10", "num_labelled_effective": 10,
           "engine": "hf", "decoding": "free",
           "f1_macro": "", "f1_prompting": "", "f1_macro_icl": "", **metrics}
    pd.DataFrame([row]).to_parquet(path / "summary.parquet")


class TestInstructionTuningCurves(unittest.TestCase):

    def test_it_is_plotted_zero_shot_and_with_icl(self):
        root = tempfile.mkdtemp()
        _summary(root, "finetuning", "finetuning_bert_base", "bert-base-uncased", f1_macro=0.5)
        _summary(root, "instruction_tuning", "instruction_tuning_qwen_9b", "Qwen/Qwen3.5-9B",
                 f1_prompting=0.8, f1_macro_icl=0.6)

        f1 = pbe.load_all(Path(root)).set_index("line")["f1"]

        self.assertEqual(f1["bert/finetuning"], 0.5)
        self.assertEqual(f1["qwen/instruction_tuning"], 0.8)
        self.assertEqual(f1["qwen/instruction_tuning_icl"], 0.6)
        self.assertEqual(pbe.pretty_label("qwen/instruction_tuning_icl"),
                         "Instruction Tuning + ICL (Qwen)")


if __name__ == "__main__":
    unittest.main()
