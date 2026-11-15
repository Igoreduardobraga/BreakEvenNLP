#!/usr/bin/env python3
"""Break-even plots: labelled samples N (log x axis) against the mean macro-F1
across folds (band: +- std), one line per model x approach, one panel per
dataset plus a combined figure.

Rules:
- Metric: finetuning, prompting and icl use f1_macro. Instruction tuning gives
  two lines: instruction_tuning -> f1_prompting (zero-shot, the format it is
  trained on) and instruction_tuning_icl -> f1_macro_icl (the same model
  evaluated with ICL). No artificial adjustments (offsets) to any curve.
- Line key: model[@variant]/approach, e.g. qwen@vllm-guided/icl.
- Incomplete folds: a warning for each point whose fold count differs from the
  dataset's mode.
- Output: aggregated CSV and PNGs in analysis/figures/.

Style (as in the POC I plots):
- Y in % (clipped to 0-100, ylim 0-105), log x axis, dashed grid, legend at the
  lower right, title DATASET.
- Break-even: first point where the best fine-tuning line rises above the best
  line of each other approach, interpolated in log space; grey vertical line
  and X marker; crossings at N <= 10 are ignored.
- Combined figure with 2 columns plus one 9x6 PNG per dataset.

Without --best-only every model is drawn: colour = approach, dash and marker =
model. The best model of each approach is a thick solid line with a star in
the legend; the others are thin and faded. It also writes
break_even_by_model[_<dataset>].png, one column per LLM against the best
fine-tuning. With --best-only there is one line per approach, as in the POC I;
use another --out-dir so the two modes do not overwrite each other's PNGs.

Usage:
    .venv/bin/python analysis/plot_break_even.py [--results-dir results_legacy]
        [--out-dir analysis/figures] [--datasets sst2 boolq]
        [--best-only] [--no-combined] [--no-single] [--no-by-model]
"""
import argparse
import logging
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompter import PromptFormatter  # noqa: E402
from run_experiments import DATASET_CONFIGS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("plot_break_even")

N_RE = re.compile(r"num_samples_(\d+)")

APPROACH_PRETTY = {
    "finetuning": "Fine-tuning",
    "icl": "In-context Learning",
    "prompting": "Prompting",
    "instruction_tuning": "Instruction Tuning",
    "instruction_tuning_steps": "Instruction Tuning",
    "instruction_tuning_icl": "Instruction Tuning + ICL",
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
    "instruction_tuning_icl": {"color": "gray", "linestyle": (0, (3, 1, 1, 1, 1, 1))},
}

APPROACH_ORDER = {
    "finetuning": 0,
    "icl": 1,
    "prompting": 2,
    "instruction_tuning": 3,
    "instruction_tuning_steps": 3,
    "instruction_tuning_icl": 4,
}

APPROACH_COLOR = {
    "finetuning": "#1f77b4",
    "icl": "#ff7f0e",
    "prompting": "#2ca02c",
    "instruction_tuning": "#d62728",
    "instruction_tuning_steps": "#d62728",
    "instruction_tuning_icl": "#9467bd",
}

# Style per model; the fixed order keeps it stable across panels and datasets.
MODEL_ORDER = ("bert", "roberta", "flan-t5", "llama2", "llama3", "mistral",
               "zephyr", "qwen", "phi", "gemma", "deberta", "chatgpt")
STYLE_COMBOS = (("-", "o"), ("--", "s"), (":", "D"), ("-.", "^"),
                ("-", "v"), ("--", "P"), (":", "X"), ("-.", "d"))


def style_for(line: str):
    """(colour, linestyle, marker): colour from the approach, style from the model."""
    exp, model = exp_of(line), model_of(line)
    color = APPROACH_COLOR.get(exp, "#7f7f7f")
    try:
        idx = MODEL_ORDER.index(model)
    except ValueError:
        idx = abs(hash(model)) % len(STYLE_COMBOS)
    ls, mk = STYLE_COMBOS[idx % len(STYLE_COMBOS)]
    return color, ls, mk


IT_TYPES = ("instruction_tuning", "instruction_tuning_steps")


def metric_for(row):
    if row["experiment_type"] in IT_TYPES:
        return row["f1_prompting"]
    if row["experiment_type"] == "instruction_tuning_icl":
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
    model_part, exp = line.split("/", 1) if "/" in line else (line, "")
    model_short, _, variant = model_part.partition("@")
    approach = APPROACH_PRETTY.get(exp, exp)
    model = MODEL_PRETTY.get(model_short, model_short)
    if variant:
        model = f"{model}, {variant}"
    return f"{approach} ({model})" if approach else model


def exp_of(line: str) -> str:
    return line.split("/", 1)[1] if "/" in line else line


def model_of(line: str) -> str:
    """Model without the variant suffix: 'qwen@vllm-guided/icl' -> 'qwen'."""
    return line.split("/", 1)[0].partition("@")[0]


def variant_of(engine, decoding) -> str:
    """'' for hf/free (the default), else '@vllm-guided' etc."""
    parts = [str(engine)] if engine != "hf" else []
    if decoding != "free":
        parts.append(str(decoding))
    return "@" + "-".join(parts) if parts else ""


@lru_cache(maxsize=None)
def train_fold_class_counts(ds: str):
    """Samples per class in a training fold (the stratified RSKF varies by +-1)."""
    try:
        # Lazy: data imports torch/transformers (~2 min cold), only needed without a recorded N.
        from data import load_text_and_targets
        _, targets, classes = load_text_and_targets(ds, 0)
    except Exception as e:  # no network/cache: fall back to the formula without a cap
        log.warning("%s: contagem por classe indisponivel (%s); N efetivo sem teto", ds, e)
        return None
    splits = DATASET_CONFIGS.get(ds, {}).get("rskf_splits", 5)
    counts = np.bincount(np.asarray(targets), minlength=len(classes))
    return tuple(int(c) for c in counts * (splits - 1) // splits)


def estimate_effective_n(n: int, ds: str) -> int:
    from sample_pool import effective_num_labelled  # imports torch (~20 s), see train_fold_class_counts
    k = len(PromptFormatter.DEFAULT_CLASSES.get(ds, ()))
    return effective_num_labelled(n, k, train_fold_class_counts(ds))


def find_summaries(results_dir: Path) -> list:
    """summary.parquet files of a results tree.

    The layout is <type>/<exp_dir>/<configuration>/<dataset>/<factor>/, so a
    fixed-depth glob finds them without walking the thousands of
    repeat_*_fold_* directories below, which takes ~30 s on a network disk.
    Falls back to a full walk for a tree with another layout.
    """
    found = sorted(results_dir.glob("*/*/*/*/*/summary.parquet"))
    return found or sorted(results_dir.rglob("summary.parquet"))


def _read_summary(p: Path):
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        log.warning("ignorando %s: %s", p, e)
        return None
    # model tag from the {exp}_{model}_{size} directory
    try:
        exp_dir = p.parents[3].name
    except IndexError:
        exp_dir = ""
    return df.assign(exp_dir=exp_dir)


def load_all(results_dir: Path) -> pd.DataFrame:
    # Reads are latency-bound on a network disk: a few threads overlap them.
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = [df for df in pool.map(_read_summary, find_summaries(results_dir)) if df is not None]
    if not frames:
        raise SystemExit(f"nenhum summary.parquet em {results_dir}")
    full = pd.concat(frames, ignore_index=True)
    # The ICL evaluation of each IT fold becomes the instruction_tuning_icl line.
    it_icl = full[full["experiment_type"].isin(IT_TYPES)].assign(experiment_type="instruction_tuning_icl")
    full = pd.concat([full, it_icl], ignore_index=True)
    # x axis = effective N (ceil(N/classes) per class, capped by each class's size):
    # recorded per fold in new results, estimated for old ones.
    full["n_nominal"] = full["configuration"].astype(str).str.extract(N_RE)[0].astype(int)
    recorded = (pd.to_numeric(full["num_labelled_effective"], errors="coerce")
                if "num_labelled_effective" in full.columns else pd.Series(np.nan, index=full.index))
    # The estimate loads the dataset: only for rows without a recorded N, once per (dataset, N).
    missing = full.loc[recorded.isna(), ["dataset", "n_nominal"]].drop_duplicates()
    estimates = {(ds, n): estimate_effective_n(n, ds) for ds, n in missing.itertuples(index=False)}
    estimated = pd.Series([estimates.get((ds, n)) for ds, n in zip(full["dataset"], full["n_nominal"])],
                          index=full.index, dtype=float)
    full["n_effective"] = recorded.fillna(estimated)
    full["n"] = (full.groupby(["dataset", "exp_dir", "experiment_type", "n_nominal"])["n_effective"]
                 .transform("mean").round().astype(int))
    full["f1"] = full.apply(metric_for, axis=1)
    full["f1"] = pd.to_numeric(full["f1"], errors="coerce")
    full = full.dropna(subset=["n", "f1"])
    full["model_short"] = [short_model(e, m) for e, m in
                           zip(full["exp_dir"], full["model"])]
    # Old summaries have no engine/decoding: they were always hf/free.
    for col, default in (("engine", "hf"), ("decoding", "free")):
        if col not in full.columns:
            full[col] = default
        full[col] = full[col].replace("", np.nan).fillna(default)
    full["variant"] = [variant_of(e, d) for e, d in zip(full["engine"], full["decoding"])]
    full["line"] = (full["model_short"] + full["variant"] + "/"
                    + full["experiment_type"].astype(str))
    return full


def aggregate(full: pd.DataFrame) -> pd.DataFrame:
    agg = (full.groupby(["dataset", "line", "n", "n_nominal"])["f1"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "f1_mean", "std": "f1_std", "count": "n_folds"}))
    agg["f1_std"] = agg["f1_std"].fillna(0.0)
    return agg


def filter_best(agg: pd.DataFrame) -> pd.DataFrame:
    """One best line per (dataset, approach): the highest mean f1_mean over the Ns."""
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
    """Crossing on a log x axis (as in the POC I).

    Interpolates y over log10(x) on a dense grid and returns the first point
    where y1 (fine-tuning) rises above y2, or None.
    """
    x1, y1 = np.array(x1, dtype=float), np.array(y1, dtype=float)
    x2, y2 = np.array(x2, dtype=float), np.array(y2, dtype=float)
    if len(x1) < 2 or len(x2) < 2:
        return None
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
    """Best line of an approach in the panel (highest mean)."""
    sub = g[g["line"].map(exp_of) == exp]
    if sub.empty:
        return None
    means = sub.groupby("line")["f1_mean"].mean()
    return means.idxmax()


def _series_for(g: pd.DataFrame, line: str):
    """Series (xs, ys in %, std in %) of a line.

    Prompting runs at a single N (no sample variation): its real value is
    repeated across the panel's x axis as a flat line, without offsets.
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


def plot_panel(ax, g: pd.DataFrame, legend_fontsize: int, detailed: bool = False) -> None:
    # Curves in % (the CSV keeps fractions).
    best_lines = ({_best_line_of(g, exp) for exp in g["line"].map(exp_of).unique()}
                  if detailed else set())
    curve_handles = []
    for line, lg in sorted(g.groupby("line")):
        xs, ys, st = _series_for(g, line)
        color, ls, mk = style_for(line)
        label = pretty_label(line)
        if not detailed:
            ls, mk, size, width, alpha, is_best = "-", "o", 4, 1.5, 1.0, True
        else:
            is_best = line in best_lines
            size, width, alpha = (5, 2.4, 1.0) if is_best else (3, 1.0, 0.55)
            if is_best:  # solid: the model is told apart by marker and legend
                ls = "-"
                label += " ★"
        (ln,) = ax.plot(xs, ys, marker=mk, markersize=size, linestyle=ls, linewidth=width,
                        alpha=alpha, color=color, label=label)
        if is_best:
            lower = np.maximum(0.0, ys - st)
            upper = np.minimum(100.0, ys + st)
            ax.fill_between(xs, lower, upper, alpha=0.15, color=color)
        curve_handles.append((APPROACH_ORDER.get(exp_of(line), 99), label, ln))

    # Break-even: best FT against the best of each other approach
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
            if bx <= 10:  # skip premature break-evens (noise)
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


def plot_singles(agg: pd.DataFrame, out_dir: Path, dpi: int, detailed: bool = False) -> None:
    for ds, g in sorted(agg.groupby("dataset")):
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_panel(ax, g, legend_fontsize=7 if detailed else 8, detailed=detailed)
        ax.set_xlabel("Amostras rotuladas (log)")
        ax.set_title(ds.upper())
        fig.tight_layout()
        out = out_dir / f"break_even_{ds}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        log.info("salvo %s", out)


def plot_combined(agg: pd.DataFrame, out_dir: Path, dpi: int,
                  name: str = "break_even_combined.png", detailed: bool = False) -> None:
    datasets = sorted(agg["dataset"].unique().tolist())
    n = len(datasets)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             squeeze=False)
    axes = axes.flatten()
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        plot_panel(ax, agg[agg["dataset"] == ds], legend_fontsize=8 if detailed else 10,
                   detailed=detailed)
        if idx // cols == rows - 1:
            ax.set_xlabel("Amostras rotuladas (log)")
        ax.set_title(ds.upper())
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    out = out_dir / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    log.info("salvo %s", out)


def llm_models(agg: pd.DataFrame) -> list:
    """Models with any approach besides fine-tuning, in the fixed order."""
    found = {model_of(ln) for ln in agg["line"] if exp_of(ln) != "finetuning"}
    return sorted(found, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER))


def model_panel_data(g: pd.DataFrame, model: str) -> pd.DataFrame:
    """Lines of one model (every approach and variant) plus the dataset's best
    fine-tuning, the reference of each panel."""
    ft_best = _best_line_of(g, "finetuning")
    keep = [ln == ft_best or (model_of(ln) == model and exp_of(ln) != "finetuning")
            for ln in g["line"]]
    return g[keep]


def plot_by_model(agg: pd.DataFrame, out_dir: Path, dpi: int, single: bool = True,
                  combined: bool = True, name: str = "break_even_by_model.png") -> None:
    """One column per LLM: the model's approaches against the dataset's best
    fine-tuning (the same reference in every column), with each approach's
    break-even. One figure per dataset and, if combined, a grid with all."""
    models = llm_models(agg)
    datasets = sorted(agg["dataset"].unique().tolist())
    if not models:
        log.warning("--by-model: nenhum modelo alem do fine-tuning nos resultados")
        return

    def draw(ax, ds, model):
        g = model_panel_data(agg[agg["dataset"] == ds], model)
        plot_panel(ax, g, legend_fontsize=7)
        ax.set_title(f"{ds.upper()} — {MODEL_PRETTY.get(model, model)}")

    if single:
        for ds in datasets:
            fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5),
                                     squeeze=False, sharey=True)
            for ax, model in zip(axes[0], models):
                draw(ax, ds, model)
                ax.set_xlabel("Amostras rotuladas (log)")
            fig.tight_layout()
            out = out_dir / f"break_even_by_model_{ds}.png"
            fig.savefig(out, dpi=dpi)
            plt.close(fig)
            log.info("salvo %s", out)
    if combined:
        fig, axes = plt.subplots(len(datasets), len(models),
                                 figsize=(5.5 * len(models), 4.2 * len(datasets)),
                                 squeeze=False, sharey=True)
        for r, ds in enumerate(datasets):
            for c, model in enumerate(models):
                draw(axes[r][c], ds, model)
                if r == len(datasets) - 1:
                    axes[r][c].set_xlabel("Amostras rotuladas (log)")
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
                    help="one best line per approach (highest mean F1 over the Ns)")
    ap.add_argument("--no-combined", action="store_true", help="skip the 2-column combined figure")
    ap.add_argument("--no-single", action="store_true", help="skip the per-dataset PNGs")
    ap.add_argument("--no-by-model", action="store_true",
                    help="skip the per-model figures (break_even_by_model*.png; "
                         "they are not written with --best-only either)")
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
    detailed = not args.best_only
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    agg.to_csv(Path(args.out_dir) / "break_even_agg.csv", index=False)
    log.info("salvo %s (%d linhas)",
             Path(args.out_dir) / "break_even_agg.csv", len(agg))
    if not args.no_single:
        plot_singles(agg, Path(args.out_dir), dpi=args.dpi_single, detailed=detailed)
    if not args.no_combined:
        plot_combined(agg, Path(args.out_dir), dpi=args.dpi_combined,
                      name=args.combined_name, detailed=detailed)
    if detailed and not args.no_by_model:
        plot_by_model(agg, Path(args.out_dir), dpi=args.dpi_single,
                      single=not args.no_single, combined=not args.no_combined)


if __name__ == "__main__":
    sys.exit(main())
