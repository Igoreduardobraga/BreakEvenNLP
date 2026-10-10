#!/usr/bin/env python3
"""Grafo principal da POC II: N (labelled samples, eixo x log) x F1-macro
medio +- desvio entre folds (banda), uma linha por (modelo x abordagem),
um painel por dataset + figura combinada.

Regras assentadas (preservadas):
- Metrica: finetuning/prompting/icl -> f1_macro; instruction_tuning -> f1_macro_icl.
  Sem forja: nenhum ajuste artificial (+/- offsets) copiado do script da POC I.
- Rotulo interno da linha: (modelo x abordagem), ex. mistral/it.
- Folds incompletos: warning no log (moda de n_folds por dataset).
- Saida: CSV agregado em analysis/figures/ + PNGs.

Estilo herdado de plotar_grafico_atualizado.py (POC I):
- Y em % (x100, clip 0-100, ylim 0-105), ylabel "F1 macro (%)".
- marker "o" markersize 4, fill_between alpha 0.15.
- xscale log, xlabel "Amostras rotuladas (log)", titulo DATASET.upper().
- grid both "--" alpha 0.3, legenda lower right.
- Break-even: intersecao FT vs cada outra abordagem em espaco log
  (geomspace 5000), vlines cinza + marcador X, filtro bx > 10.
- Figura combinada 2-colunas (14x10 p/ 2x2) + PNGs individuais 9x6.

Uso:
    .venv/bin/python analysis/plot_break_even.py [--results-dir results_legacy]
        [--out-dir analysis/figures] [--datasets sst2 boolq]
        [--best-only] [--no-combined] [--no-single]
"""
import argparse
import logging
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("plot_break_even")

N_RE = re.compile(r"num_samples_(\d+)")

APPROACH_PRETTY = {
    "finetuning": "Fine-tuning",
    "icl": "In-context Learning",
    "prompting": "Prompting",
    "instruction_tuning": "Instruction Tuning",
    "instruction_tuning_steps": "Instruction Tuning",
}

MODEL_PRETTY = {
    "flan-t5": "FLAN-T5-base",
    "llama2": "LLaMA-2-13B",
    "llama3": "LLaMA-3-8B",
    "mistral": "Mistral-7B",
    "zephyr": "Zephyr-7B",
    "bert": "BERT-base",
    "roberta": "RoBERTa-base",
    "qwen": "Qwen",
    "phi": "Phi",
    "gemma": "Gemma",
    "deberta": "DeBERTa",
    "chatgpt": "ChatGPT",
}

BREAK_EVEN_CONFIG = {
    "prompting": {"color": "gray", "linestyle": "--"},
    "icl": {"color": "gray", "linestyle": ":"},
    "instruction_tuning": {"color": "gray", "linestyle": "-."},
    "instruction_tuning_steps": {"color": "gray", "linestyle": "-."},
}

# Ordem fixa da legenda por abordagem.
APPROACH_ORDER = {
    "finetuning": 0,
    "icl": 1,
    "prompting": 2,
    "instruction_tuning": 3,
    "instruction_tuning_steps": 3,
}

# Cor fixa por abordagem (C0-C3 do ciclo tab10).
APPROACH_COLOR = {
    "finetuning": "#1f77b4",
    "icl": "#ff7f0e",
    "prompting": "#2ca02c",
    "instruction_tuning": "#d62728",
    "instruction_tuning_steps": "#d62728",
}

# Estilo por modelo (ordem fixa p/ estabilidade entre paineis/datasets).
MODEL_ORDER = ("bert", "roberta", "flan-t5", "llama2", "llama3", "mistral",
               "zephyr", "qwen", "phi", "gemma", "deberta", "chatgpt")
STYLE_COMBOS = (("-", "o"), ("--", "s"), (":", "D"), ("-.", "^"),
                ("-", "v"), ("--", "P"), (":", "X"), ("-.", "d"))


def style_for(line: str):
    """(cor, linestyle, marker): cor pela abordagem, estilo pelo modelo."""
    exp, model = exp_of(line), line.split("/", 1)[0] if "/" in line else line
    color = APPROACH_COLOR.get(exp, "#7f7f7f")
    try:
        idx = MODEL_ORDER.index(model)
    except ValueError:
        idx = abs(hash(model)) % len(STYLE_COMBOS)
    ls, mk = STYLE_COMBOS[idx % len(STYLE_COMBOS)]
    return color, ls, mk


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


def pretty_label(line: str) -> str:
    """'mistral/instruction_tuning' -> 'Instruction Tuning (Mistral-7B)'."""
    if "/" in line:
        model_short, exp = line.split("/", 1)
    else:
        model_short, exp = line, ""
    approach = APPROACH_PRETTY.get(exp, exp)
    model = MODEL_PRETTY.get(model_short, model_short)
    return f"{approach} ({model})" if approach else model


def exp_of(line: str) -> str:
    return line.split("/", 1)[1] if "/" in line else line


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


def filter_best(agg: pd.DataFrame) -> pd.DataFrame:
    """Uma BestLine por (dataset, abordagem): maior media de f1_mean nos Ns."""
    agg = agg.copy()
    agg["exp"] = agg["line"].map(exp_of)
    mean_by_line = agg.groupby(["dataset", "exp", "line"])["f1_mean"].mean().reset_index()
    best_idx = mean_by_line.groupby(["dataset", "exp"])["f1_mean"].idxmax()
    best_lines = set(mean_by_line.loc[best_idx, ["dataset", "line"]].itertuples(index=False, name=None))
    keep = [(d, ln) in best_lines for d, ln in zip(agg["dataset"], agg["line"])]
    out = agg[keep].drop(columns=["exp"]).reset_index(drop=True)
    for ds, g in out.groupby("dataset"):
        log.info("%s --best-only: %s", ds, sorted(g["line"].unique().tolist()))
    return out


def warn_incomplete(agg: pd.DataFrame) -> None:
    for ds, g in agg.groupby("dataset"):
        mode_n = int(g["n_folds"].mode().iloc[0])
        bad = g[g["n_folds"] != mode_n]
        for _, r in bad.iterrows():
            log.warning("%s: linha %s N=%d tem %d folds (moda=%d)",
                        ds, r["line"], int(r["n"]), int(r["n_folds"]), mode_n)


def find_intersection(x1, y1, x2, y2):
    """Intersecao em escala LOG no eixo X (igual a POC I).

    Interpola Y sobre log10(X) num grid denso e retorna o primeiro ponto
    onde y1 (fine-tuning) supera y2, ou None.
    """
    x1, y1 = np.array(x1, dtype=float), np.array(y1, dtype=float)
    x2, y2 = np.array(x2, dtype=float), np.array(y2, dtype=float)
    if len(x1) < 2 or len(x2) < 2:
        return None
    # remove NaN e ordena
    m1 = np.isfinite(x1) & np.isfinite(y1) & (x1 > 0)
    m2 = np.isfinite(x2) & np.isfinite(y2) & (x2 > 0)
    x1, y1 = x1[m1], y1[m1]
    x2, y2 = x2[m2], y2[m2]
    if len(x1) < 2 or len(x2) < 2:
        return None
    o1, o2 = np.argsort(x1), np.argsort(x2)
    x1, y1, x2, y2 = x1[o1], y1[o1], x2[o2], y2[o2]

    min_x = max(np.min(x1), np.min(x2))
    max_x = min(np.max(x1), np.max(x2))
    if min_x >= max_x:
        return None

    x_common = np.geomspace(min_x, max_x, 5000)
    y1_interp = np.interp(np.log10(x_common), np.log10(x1), y1)
    y2_interp = np.interp(np.log10(x_common), np.log10(x2), y2)
    diff = y1_interp - y2_interp
    for i in range(len(diff) - 1):
        if diff[i] <= 0 and diff[i + 1] > 0:
            return x_common[i], y1_interp[i]
    return None


def _best_line_of(g: pd.DataFrame, exp: str):
    """Melhor linha de uma abordagem dentro do painel (maior media)."""
    sub = g[g["line"].map(exp_of) == exp]
    if sub.empty:
        return None
    means = sub.groupby("line")["f1_mean"].mean()
    return means.idxmax()


def _series_for(g: pd.DataFrame, line: str):
    """Serie (xs, ys%, std%) de uma linha.

    Prompting foi executado num unico N (sem variacao de sample):
    repete o ponto linearmente por todo o eixo X do painel como reta
    continua. Sem forja: usa o valor real, sem offsets.
    """
    lg = g[g["line"] == line].sort_values("n")
    if exp_of(line) == "prompting" and lg["n"].nunique() == 1:
        all_xs = np.array(sorted(g["n"].unique().tolist()), dtype=float)
        y0 = float(lg["f1_mean"].iloc[0]) * 100.0
        s0 = float(lg["f1_std"].iloc[0]) * 100.0
        return all_xs, np.full_like(all_xs, y0), np.full_like(all_xs, s0)
    return (lg["n"].to_numpy(dtype=float),
            lg["f1_mean"].to_numpy(dtype=float) * 100.0,
            lg["f1_std"].to_numpy(dtype=float) * 100.0)


def plot_panel(ax, g: pd.DataFrame, legend_fontsize: int) -> None:
    # Curvas em % (preserva CSV em fracao; so o visual converte).
    # Cor fixa por abordagem; linestyle/marker uniformes ("-", "o") por enquanto.
    # Vlines seguem cinzas.
    curve_handles = []
    for line, lg in sorted(g.groupby("line")):
        xs, ys, st = _series_for(g, line)
        color, _, _ = style_for(line)
        (ln,) = ax.plot(xs, ys, marker="o", markersize=4, linestyle="-",
                        color=color, label=pretty_label(line))
        lower = np.maximum(0.0, ys - st)
        upper = np.minimum(100.0, ys + st)
        ax.fill_between(xs, lower, upper, alpha=0.15, color=color)
        curve_handles.append((APPROACH_ORDER.get(exp_of(line), 99), pretty_label(line), ln))

    # Break-even: melhor FT vs melhor de cada outra abordagem
    ft_best = _best_line_of(g, "finetuning")
    if ft_best is not None:
        ft_x, ft_y, _ = _series_for(g, ft_best)
        for other_exp in sorted(g["line"].map(exp_of).unique()):
            if other_exp == "finetuning":
                continue
            other_best = _best_line_of(g, other_exp)
            if other_best is None:
                continue
            ot_x, ot_y, _ = _series_for(g, other_best)
            inter = find_intersection(ft_x, ft_y, ot_x, ot_y)
            if inter is None:
                continue
            bx, by = float(inter[0]), float(inter[1])
            if bx <= 10:  # filtra break-evens prematuros (ruido)
                continue
            cfg = BREAK_EVEN_CONFIG.get(other_exp, {"color": "black", "linestyle": "--"})
            ax.vlines(x=bx, ymin=0, ymax=by, colors=cfg["color"],
                      linestyles=cfg["linestyle"], linewidth=1.5, alpha=0.8)
            ax.plot(bx, by, marker="X", color=cfg["color"], markersize=8, zorder=10)
            log.info("  [BREAK-EVEN] FT (%s) igualou %s (%s) em N=%.1f (F1=%.1f%%)",
                     ft_best, other_exp, other_best, bx, by)

    ax.set_xscale("log")
    ax.set_ylabel("F1 macro (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    curve_handles.sort(key=lambda t: (t[0], t[1]))
    ax.legend([h for _, _, h in curve_handles],
              [lb for _, lb, _ in curve_handles],
              fontsize=legend_fontsize, loc="lower right")


def plot_singles(agg: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    for ds, g in sorted(agg.groupby("dataset")):
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_panel(ax, g, legend_fontsize=8)
        ax.set_xlabel("Amostras rotuladas (log)")
        ax.set_title(ds.upper())
        fig.tight_layout()
        out = out_dir / f"break_even_{ds}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        log.info("salvo %s", out)


def plot_combined(agg: pd.DataFrame, out_dir: Path, dpi: int,
                  name: str = "break_even_combined.png") -> None:
    datasets = sorted(agg["dataset"].unique().tolist())
    n = len(datasets)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             squeeze=False)
    axes = axes.flatten()
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        plot_panel(ax, agg[agg["dataset"] == ds], legend_fontsize=10)
        if idx // cols == rows - 1:  # so a linha de baixo tem xlabel
            ax.set_xlabel("Amostras rotuladas (log)")
        ax.set_title(ds.upper())
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    out = out_dir / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    log.info("salvo %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="analysis/figures")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--best-only", action="store_true",
                    help="uma BestLine por abordagem (maior media de F1 nos Ns)")
    ap.add_argument("--no-combined", action="store_true", help="nao gera figura 2-colunas")
    ap.add_argument("--no-single", action="store_true", help="nao gera PNGs individuais")
    ap.add_argument("--dpi-combined", type=int, default=300)
    ap.add_argument("--dpi-single", type=int, default=150)
    ap.add_argument("--combined-name", default="break_even_combined.png")
    args = ap.parse_args()

    full = load_all(Path(args.results_dir))
    if args.datasets:
        full = full[full["dataset"].isin(args.datasets)]
    agg = aggregate(full)
    warn_incomplete(agg)
    if args.best_only:
        agg = filter_best(agg)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    agg.to_csv(Path(args.out_dir) / "break_even_agg.csv", index=False)
    log.info("salvo %s (%d linhas)",
             Path(args.out_dir) / "break_even_agg.csv", len(agg))
    if not args.no_single:
        plot_singles(agg, Path(args.out_dir), dpi=args.dpi_single)
    if not args.no_combined:
        plot_combined(agg, Path(args.out_dir), dpi=args.dpi_combined,
                      name=args.combined_name)


if __name__ == "__main__":
    sys.exit(main())
