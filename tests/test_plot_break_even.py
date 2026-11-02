"""Instruction tuning gives two curves in the break-even plot; several models per approach stay readable."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# The plots run from .venv; .venv-vllm has no matplotlib.
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    from analysis import plot_break_even as pbe


def _summary(root, exp_type, exp_dir, model, **metrics):
    path = Path(root) / exp_type / exp_dir / "num_samples_10" / "toy" / "golden_model"
    path.mkdir(parents=True)
    row = {"experiment_type": exp_type, "model": model, "dataset": "toy",
           "configuration": "num_samples_10", "num_labelled_effective": 10,
           "engine": "hf", "decoding": "free",
           "f1_macro": "", "f1_prompting": "", "f1_macro_icl": "", **metrics}
    pd.DataFrame([row]).to_parquet(path / "summary.parquet")


@unittest.skipUnless(HAS_MATPLOTLIB, "needs matplotlib")
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


def _agg(lines):
    rows = [("toy", line, n, n, 0.5 + 0.05 * i + 0.02 * k, 0.01, 5)
            for i, line in enumerate(lines) for k, n in enumerate((14, 100, 1000))]
    return pd.DataFrame(rows, columns=["dataset", "line", "n", "n_nominal", "f1_mean", "f1_std", "n_folds"])


LINES = ["bert/finetuning", "deberta/finetuning", "qwen/prompting", "phi/prompting",
         "qwen/icl", "phi/icl", "qwen/instruction_tuning", "qwen/instruction_tuning_icl"]


@unittest.skipUnless(HAS_MATPLOTLIB, "needs matplotlib")
class TestManyModelsPerApproach(unittest.TestCase):

    def _plot(self, detailed):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        pbe.plot_panel(ax, _agg(LINES), legend_fontsize=7, detailed=detailed)
        self.addCleanup(plt.close, fig)
        return {ln.get_label(): ln for ln in ax.get_lines() if ln.get_label()[0] != "_"}

    def test_color_is_the_approach_and_the_model_is_the_marker(self):
        lines = self._plot(detailed=True)
        by_prefix = lambda prefix: next(ln for label, ln in lines.items() if label.startswith(prefix))
        qwen, phi = by_prefix("In-context Learning (Qwen)"), by_prefix("In-context Learning (Phi)")
        self.assertEqual(qwen.get_color(), phi.get_color())
        self.assertEqual(qwen.get_color(), pbe.APPROACH_COLOR["icl"])
        self.assertNotEqual(qwen.get_marker(), phi.get_marker())
        self.assertEqual(by_prefix("Instruction Tuning + ICL (Qwen)").get_color(),
                         pbe.APPROACH_COLOR["instruction_tuning_icl"])  # purple

    def test_only_the_best_model_of_each_approach_is_emphasised(self):
        lines = self._plot(detailed=True)
        best = [label for label, ln in lines.items() if ln.get_linewidth() > 2]
        self.assertEqual(sorted(best), sorted(label for label in lines if label.endswith("\u2605")))
        # the best fine-tuning is DeBERTa and the best prompting is Phi (highest means in _agg)
        self.assertIn("Fine-tuning (DeBERTa) \u2605", best)
        self.assertIn("Prompting (Phi) \u2605", best)
        self.assertEqual(len(best), 5)  # one per approach: FT, prompting, ICL, IT, IT + ICL

    def test_best_only_panel_keeps_the_original_look(self):
        for label, ln in self._plot(detailed=False).items():
            self.assertEqual((ln.get_marker(), ln.get_linestyle(), ln.get_linewidth()), ("o", "-", 1.5), label)
            self.assertNotIn("\u2605", label)

    def test_by_model_panels_hold_one_model_and_the_best_finetuning(self):
        agg = _agg(LINES)
        self.assertEqual(pbe.llm_models(agg), ["qwen", "phi"])
        panel = pbe.model_panel_data(agg, "phi")
        self.assertEqual(sorted(panel["line"].unique()), ["deberta/finetuning", "phi/icl", "phi/prompting"])

    def test_by_model_figures_are_written(self):
        out = Path(tempfile.mkdtemp())
        pbe.plot_by_model(_agg(LINES), out, dpi=50)
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["break_even_by_model.png", "break_even_by_model_toy.png"])


if __name__ == "__main__":
    unittest.main()
