#!/usr/bin/env python3
"""Evidence for the BoolQ truncation fix in fine-tuning.

Runs the same BoolQ fine-tuning twice, on the same folds and seeds:
- pair: the current encoding, a (question, passage) pair that cuts only the
  passage;
- legacy: a single "Passage ... Question ..." text cut at the end, as before
  (the question is lost in ~45% of the samples at max_len=128).

The encoding is the only difference: legacy runs main.py in the same process
with sample_pool.split_text_pair disabled. Compares the test F1 (common
1000-sample subset) fold by fold with a paired Wilcoxon test. Resumes where it
stopped: folds already saved are skipped.

Usage (from the repository root):
    uv run python analysis/check_boolq_truncation.py \
        [--models roberta deberta] [--n 5000] [--lr 3e-5] [--gpus 0]
"""
import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from result_store import experiment_dir_name  # noqa: E402
from run_experiments import finetuning_experiment  # noqa: E402

VARIANTS = ("pair", "legacy")


def worker(variant, main_args):
    """Runs main.py in this process; the legacy variant turns the pair encoding off."""
    import sample_pool
    if variant == "legacy":
        sample_pool.split_text_pair = lambda text: None
    sys.argv = ["main.py", *main_args]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def summary_path(results_root, model, n):
    return (Path(results_root) / "finetuning" / experiment_dir_name("finetuning", model, "base")
            / f"num_samples_{n}" / "boolq" / "golden_model" / "summary.parquet")


def run(args):
    for model in args.models:
        for variant in VARIANTS:
            root = os.path.join(args.results_root, variant)
            exp = finetuning_experiment("boolq", model, args.n, args.lr, rskf_repeats=1, results_root=root)
            main_args = exp["cmd_args"][2:]  # without "python main.py"
            log = Path(args.results_root) / "logs" / f"{model}_{variant}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if args.gpus is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.gpus
            print(f"[{model} / {variant}] N={args.n} lr={args.lr:g} -> {log}", flush=True)
            with open(log, "a") as f:
                code = subprocess.call([sys.executable, __file__, "--worker", variant, "--", *main_args],
                                       cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if code != 0:
                print(f"  FALHOU (exit {code}), veja o log", flush=True)


def report(args):
    rows = []
    for model in args.models:
        folds = {}
        for variant in VARIANTS:
            p = summary_path(os.path.join(args.results_root, variant), model, args.n)
            if not p.exists():
                print(f"{model}/{variant}: sem resultados em {p}")
                continue
            df = pd.read_parquet(p)
            df["f1_macro"] = pd.to_numeric(df["f1_macro"], errors="coerce")
            folds[variant] = df.set_index(["rskf_repeat", "rskf_fold"])["f1_macro"]
        if len(folds) < 2:
            continue
        paired = pd.concat(folds, axis=1).dropna()
        diff = paired["pair"] - paired["legacy"]
        p_value = wilcoxon(diff).pvalue if len(diff) >= 5 and diff.abs().sum() > 0 else float("nan")
        rows.append(dict(model=model, folds=len(paired),
                         f1_legacy=paired["legacy"].mean(), f1_pair=paired["pair"].mean(),
                         diff_mean=diff.mean(), diff_std=diff.std(),
                         folds_improved=int((diff > 0).sum()), wilcoxon_p=p_value))
        print(f"\n{model}: F1 por fold")
        print(paired.assign(diff=diff).round(3).to_string())
    if not rows:
        sys.exit("nada para comparar ainda")
    table = pd.DataFrame(rows)
    out = Path(args.results_root) / "boolq_truncation_check.csv"
    table.to_csv(out, index=False)
    print(f"\nBoolQ N={args.n}, lr={args.lr:g}, pair (correcao) vs legacy (pergunta cortada):")
    print(table.round(4).to_string(index=False))
    print(f"\nsalvo em {out}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        sep = sys.argv.index("--")
        return worker(sys.argv[2], sys.argv[sep + 1:])

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["roberta", "deberta"])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--results-root", default=str(ROOT / "results_boolq_check"))
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        run(args)
    report(args)


if __name__ == "__main__":
    main()
