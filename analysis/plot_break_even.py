#!/usr/bin/env python3
"""Grafo principal da POC II: N (labelled samples, eixo x log) x F1-macro
medio +- desvio entre folds (banda), uma linha por (modelo x abordagem),
um painel por dataset.

Regras assentadas:
- Metrica: finetuning/prompting/icl -> f1_macro; instruction_tuning -> f1_macro_icl.
- Rotulo da linha: (modelo x abordagem), ex. mistral/it.
- Folds incompletos: plota com contagem de folds anotada + warning no log.
- Saida: PNG 150 DPI + CSV agregado em analysis/figures/.

Uso:
    .venv/bin/python analysis/plot_break_even.py [--results-dir results_legacy]
        [--out-dir analysis/figures] [--datasets sst2 boolq]
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("plot_break_even")

N_RE = re.compile(r"num_samples_(\d+)")


def metric_for(row):
    if row["experiment_type"] in ("instruction_tuning", "instruction_tuning_steps"):
        return row["f1_macro_icl"]
    return row["f1_macro"]


def short_model(exp_dir_model: str, model_col: str) -> str:
    exp_dir_model = exp_dir_model.lower()
    for cand in ("flan-t5", "llama3", "llama2", "mistral", "zephyr",
                 "qwen", "phi", "gemma", "roberta", "deberta", "bert", "chatgpt"):
        if cand in exp_dir_model or cand in str(model_col).lower():
            return cand
    return str(model_col)


def load_all(results_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(results_dir.rglob("summary.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            log.warning("ignorando %s: %s", p, e)
            continue
        # tag de modelo vinda do diretorio {exp}_{modelo}_{size}
        try:
            exp_dir = p.parents[3].name
        except IndexError:
            exp_dir = ""
        df = df.copy()
        df["exp_dir"] = exp_dir
        frames.append(df)
    if not frames:
        raise SystemExit(f"nenhum summary.parquet em {results_dir}")
    full = pd.concat(frames, ignore_index=True)
    full["n"] = full["configuration"].astype(str).str.extract(N_RE)[0].astype(int)
    full["f1"] = full.apply(metric_for, axis=1)
    full["f1"] = pd.to_numeric(full["f1"], errors="coerce")
    full = full.dropna(subset=["n", "f1"])
    full["model_short"] = [short_model(e, m) for e, m in
                           zip(full["exp_dir"], full["model"])]
    full["line"] = full["model_short"] + "/" + full["experiment_type"].astype(str)
    return full


def aggregate(full: pd.DataFrame) -> pd.DataFrame:
    agg = (full.groupby(["dataset", "line", "n"])["f1"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "f1_mean", "std": "f1_std", "count": "n_folds"}))
    agg["f1_std"] = agg["f1_std"].fillna(0.0)
    return agg


def warn_incomplete(agg: pd.DataFrame) -> None:
    for ds, g in agg.groupby("dataset"):
        mode_n = int(g["n_folds"].mode().iloc[0])
        bad = g[g["n_folds"] != mode_n]
        for _, r in bad.iterrows():
            log.warning("%s: linha %s N=%d tem %d folds (moda=%d)",
                        ds, r["line"], int(r["n"]), int(r["n_folds"]), mode_n)


def plot(agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, g in sorted(agg.groupby("dataset")):
        fig, ax = plt.subplots(figsize=(9, 6))
        for line, lg in sorted(g.groupby("line")):
            lg = lg.sort_values("n")
            ax.semilogx(lg["n"], lg["f1_mean"], marker="o", label=line)
            ax.fill_between(lg["n"], lg["f1_mean"] - lg["f1_std"],
                            lg["f1_mean"] + lg["f1_std"], alpha=0.2)
            for _, r in lg.iterrows():
                ax.annotate(f"n={int(r['n_folds'])}", (r["n"], r["f1_mean"]),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color="dimgray")
        ax.set_xlabel("labelled samples (N, log)")
        ax.set_ylabel("macro-F1 (mean +- std over folds)")
        ax.set_title(f"Break-even: {ds}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"break_even_{ds}.png", dpi=150)
        plt.close(fig)
        log.info("salvo %s", out_dir / f"break_even_{ds}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="analysis/figures")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()

    full = load_all(Path(args.results_dir))
    if args.datasets:
        full = full[full["dataset"].isin(args.datasets)]
    agg = aggregate(full)
    warn_incomplete(agg)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    agg.to_csv(Path(args.out_dir) / "break_even_agg.csv", index=False)
    log.info("salvo %s (%d linhas)",
             Path(args.out_dir) / "break_even_agg.csv", len(agg))
    plot(agg, Path(args.out_dir))


if __name__ == "__main__":
    sys.exit(main())
