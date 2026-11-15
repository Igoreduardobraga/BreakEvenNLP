#!/usr/bin/env python3
"""Sustainability plots: the cost of each approach per fold, measured by
SustainabilityTracker (energy_kwh, co2_kg, duration_seconds in summary.parquet).

Figures (one panel per dataset, colour = approach, marker = model, as in
plot_break_even.py):
- sustainability_cost.png: cost per fold against the labelled samples N (log-log).
  Instruction Tuning + ICL is left out: it is the same fold as Instruction Tuning.
- sustainability_tradeoff.png: macro-F1 against cost per fold; each line goes
  through the Ns in increasing order, so up-and-right means more F1 for more cost.

What a fold's cost covers: the evaluation of the 1000-sample test subset, plus
training for fine-tuning and IT and, for IT, loading the model (the IT cost
also pays for its ICL evaluation). The vLLM engine of prompting and ICL is
built once per process, outside the folds, so its load is not counted.
Fine-tuning's GPU energy was sampled with nvidia-smi; the other approaches
read NVML's energy counter. CO2 is energy times a fixed carbon intensity.

Usage:
    .venv/bin/python analysis/plot_sustainability.py [--metric energy|co2|time]
        [--results-dir results] [--out-dir analysis/figures] [--datasets sst2 boolq]
        [--best-only]
"""
import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_break_even as pbe  # noqa: E402

log = logging.getLogger("plot_sustainability")

# column, factor to the plotted unit, axis label
METRICS = {
    "energy": ("energy_kwh", 1000.0, "Energia por fold (Wh)"),
    "co2": ("co2_kg", 1000.0, "CO$_2$ por fold (g)"),
    "time": ("duration_seconds", 1 / 60, "Tempo por fold (min)"),
}


def aggregate(full: pd.DataFrame) -> pd.DataFrame:
    full = full.copy()
    for col in ("energy_kwh", "co2_kg", "duration_seconds"):
        full[col] = pd.to_numeric(full[col], errors="coerce")
    agg = (full.groupby(["dataset", "line", "n", "n_nominal"])
           .agg(f1_mean=("f1", "mean"), n_folds=("f1", "size"),
                energy_kwh=("energy_kwh", "mean"), energy_kwh_std=("energy_kwh", "std"),
                co2_kg=("co2_kg", "mean"), duration_seconds=("duration_seconds", "mean"))
           .reset_index())
    return agg


def _legend_handles(lines, best):
    handles = []
    for line in sorted(lines, key=lambda ln: (pbe.APPROACH_ORDER.get(pbe.exp_of(ln), 99), ln)):
        color, ls, mk = pbe.style_for(line)
        if best:
            ls, mk = "-", "o"
        handles.append(plt.Line2D([], [], color=color, linestyle=ls, marker=mk,
                                  markersize=5, label=pbe.pretty_label(line)))
    return handles


def _style(line, best):
    color, ls, mk = pbe.style_for(line)
    return (color, "-", "o") if best else (color, ls, mk)


def plot_cost(ax, g, col, factor, best):
    xs_all = np.array(sorted(g["n"].unique()), dtype=float)
    for line, lg in g.groupby("line"):
        if pbe.exp_of(line) == "instruction_tuning_icl":
            continue
        lg = lg.sort_values("n")
        color, ls, mk = _style(line, best)
        xs, ys = lg["n"].to_numpy(float), lg[col].to_numpy(float) * factor
        if pbe.exp_of(line) == "prompting" and len(lg) == 1:
            # No training: one cost, drawn across the panel as in the break-even plots.
            xs, ys = xs_all, np.full_like(xs_all, ys[0])
        ax.plot(xs, ys, color=color, linestyle=ls, marker=mk, markersize=4, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_tradeoff(ax, g, col, factor, best):
    for line, lg in g.groupby("line"):
        lg = lg.sort_values("n")
        color, ls, mk = _style(line, best)
        ax.plot(lg[col] * factor, lg["f1_mean"] * 100, color=color, linestyle=ls,
                marker=mk, markersize=4, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_ylabel("F1 macro (%)")


def figure(agg, draw, xlabel, ylabel, out, dpi, best, drawn=lambda line: True, legend_loc="best"):
    """drawn(line) tells which lines the panels draw, so the legend lists only those."""
    datasets = sorted(agg["dataset"].unique())
    cols = 2 if len(datasets) > 1 else 1
    rows = math.ceil(len(datasets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.6 * rows), squeeze=False)
    axes = axes.flatten()
    for ax, ds in zip(axes, datasets):
        g = agg[agg["dataset"] == ds]
        draw(ax, g)
        ax.set_title(ds.upper())
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if best:  # the best model changes per dataset: one legend per panel
            ax.legend(handles=_legend_handles([ln for ln in g["line"].unique() if drawn(ln)], best),
                      fontsize=7, loc=legend_loc)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    if best:
        fig.tight_layout()
    else:
        handles = _legend_handles([ln for ln in agg["line"].unique() if drawn(ln)], best)
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8,
                   frameon=False)
        legend_rows = math.ceil(len(handles) / min(4, len(handles)))
        fig.tight_layout(rect=(0, 0.035 * legend_rows / rows, 1, 1))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    log.info("salvo %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="analysis/figures")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--metric", default="energy", choices=list(METRICS))
    ap.add_argument("--best-only", action="store_true",
                    help="one line per approach: its best model by mean F1, as in plot_break_even.py")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    full = pbe.load_all(Path(args.results_dir))
    if args.datasets:
        full = full[full["dataset"].isin(args.datasets)]
    agg = aggregate(full)
    if args.best_only:
        agg = pbe.filter_best(agg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_best" if args.best_only else ""
    agg.to_csv(out_dir / f"sustainability_agg{suffix}.csv", index=False)

    col, factor, label = METRICS[args.metric]
    figure(agg, lambda ax, g: plot_cost(ax, g, col, factor, args.best_only),
           "Amostras rotuladas (log)", label,
           out_dir / f"sustainability_cost_{args.metric}{suffix}.png", args.dpi, args.best_only,
           drawn=lambda ln: pbe.exp_of(ln) != "instruction_tuning_icl")
    figure(agg, lambda ax, g: plot_tradeoff(ax, g, col, factor, args.best_only),
           f"{label}, log", None,
           out_dir / f"sustainability_tradeoff_{args.metric}{suffix}.png", args.dpi, args.best_only,
           legend_loc="lower right")

    # Whole-grid cost per approach: every fold summed, IT + ICL excluded (same folds as IT).
    fold_cost = full[full["line"].map(pbe.exp_of) != "instruction_tuning_icl"].copy()
    fold_cost[col] = pd.to_numeric(fold_cost[col], errors="coerce") * factor
    total = (fold_cost.assign(approach=fold_cost["line"].map(pbe.exp_of))
             .groupby("approach")[col].agg(["sum", "count"]))
    log.info("custo total da grade, soma dos folds (%s):\n%s",
             label.replace(" por fold", ""), total.round(1).to_string())


if __name__ == "__main__":
    sys.exit(main())
