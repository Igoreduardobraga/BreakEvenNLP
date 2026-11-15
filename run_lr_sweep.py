#!/usr/bin/env python3
"""Learning-rate sweep of fine-tuning or instruction tuning: one LR per model
and N band.

For each model, N band (FT_LR_BANDS in run_experiments.py), dataset and LR, it
trains at the band's probe N and picks the LR with the lowest validation loss
(10% of the training pool), by mean rank across datasets. The test F1 is kept
for reference only: choosing by it would favour fine-tuning over the LLMs.

- finetuning: repeat 0 of the RSKF; val_loss of the best epoch, or of the
  final model on a 10% holdout in the small band (no early stopping there).
- instruction_tuning: the LR of every IT model, the LoRA of the causal models
  (Qwen, Phi, Gemma, Mistral, Zephyr) and Flan-T5's full fine-tuning, which
  has its own grid; val_loss of the best checkpoint, on 2 folds per
  configuration and without test generations (--it_val_only), since each IT
  step costs far more.

Results go to --results-root (results_lr_sweep/ or results_it_lr_sweep/) and
the choice to ft_learning_rates.json or it_learning_rates.json, which
run_experiments.py reads in the finetuning and IT scripts. The file keeps the
choice of the models a run leaves out.

Usage:
    uv run python run_lr_sweep.py [--approach finetuning|instruction_tuning]
        [--models bert roberta deberta | qwen_9b phi_mini mistral_base flan-t5_base ...]
        [--datasets sst2 boolq ag_news snips] [--lrs ...]
        [--gpus 0] [--dry-run] [--select-only]
"""
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

from run_experiments import (DATASET_CONFIGS, FT_LR_BANDS, FT_LR_FILE, IT_FLAN_T5_MODELS,
                             IT_LR_FILE, IT_MISTRAL_ZEPHYR_MODELS, IT_MODERN_MODELS,
                             finetuning_experiment, is_already_completed, it_experiment,
                             it_lr_key, results_dir, run_job)

IT_MODELS = {it_lr_key(m, size): (m, size)
             for m, size in IT_MODERN_MODELS + IT_FLAN_T5_MODELS + IT_MISTRAL_ZEPHYR_MODELS}

# Flan-T5 is fully fine-tuned, not LoRA: half-decade steps from its legacy 1e-5
# up to 1e-3, the usual range for T5 with AdamW.
FLAN_T5_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

# IT's LoRA grid starts at 2e-5 because Gemma and Qwen picked the former lower edge, 5e-5.
APPROACHES = {
    "finetuning": dict(
        models=["bert", "roberta", "deberta"], lrs=[1e-5, 2e-5, 3e-5, 5e-5],
        results_root="results_lr_sweep", logs_dir=os.path.join("logs", "lr_sweep"), out=FT_LR_FILE,
        criterion="min val_loss (best epoch), mean rank across datasets"),
    "instruction_tuning": dict(
        models=list(IT_MODELS), lrs=[2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
        model_lrs={it_lr_key(*IT_FLAN_T5_MODELS[0]): FLAN_T5_LRS},
        results_root="results_it_lr_sweep", logs_dir=os.path.join("logs", "it_lr_sweep"), out=IT_LR_FILE,
        criterion="min val_loss (best checkpoint), mean rank across datasets"),
}

# Folds per IT sweep configuration (repeat 0 of a 2-split RSKF).
IT_SWEEP_SPLITS = 2


def default_lrs(approach, models):
    """LR grid per model: the approach's grid, or the model's own one."""
    defaults = APPROACHES[approach]
    return {m: defaults.get("model_lrs", {}).get(m, defaults["lrs"]) for m in models}


def sweep_experiments(models, datasets, lrs, results_root, approach="finetuning"):
    """lrs: one grid for every model, or a dict with each model's grid."""
    exps = []
    for m in models:
        for band, _, probe_n in FT_LR_BANDS:
            for ds in datasets:
                for lr in (lrs[m] if isinstance(lrs, dict) else lrs):
                    cfg_name = f"lr_{lr:g}_num_samples_{probe_n}"
                    if approach == "finetuning":
                        exp = finetuning_experiment(
                            ds, m, probe_n, lr, rskf_repeats=1, configuration_name=cfg_name,
                            results_root=results_root, holdout_val=True)
                    else:
                        exp = it_experiment(
                            ds, *IT_MODELS[m], probe_n, lr, rskf_splits=IT_SWEEP_SPLITS,
                            configuration_name=cfg_name, results_root=results_root,
                            holdout_val=True, val_only=True)
                    exp["band"] = band
                    exp["lr_key"] = m
                    exps.append(exp)
    return exps


def load_sweep_results(exps):
    rows = []
    for exp in exps:
        path = os.path.join(results_dir(exp), "summary.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        for col in ("val_loss", "val_f1_macro", "f1_macro"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        rows.append(dict(model=exp["lr_key"], band=exp["band"], n=exp["num_labelled"],
                         dataset=exp["dataset"], lr=exp["lr"], folds=len(df),
                         val_folds=int(df["val_loss"].notna().sum()),
                         val_loss=df["val_loss"].mean(), val_f1_macro=df["val_f1_macro"].mean(),
                         test_f1_macro=df["f1_macro"].mean()))
    return pd.DataFrame(rows)


def select(table):
    """Ranks the LRs by val_loss per dataset; the lowest mean rank wins (ties: lower val_loss)."""
    table = table.dropna(subset=["val_loss"]).copy()
    table["rank"] = table.groupby(["model", "band", "dataset"])["val_loss"].rank(method="min")
    score = (table.groupby(["model", "band", "lr"])
             .agg(mean_rank=("rank", "mean"), val_loss=("val_loss", "mean"),
                  datasets=("dataset", "nunique"))
             .reset_index())
    chosen = {}
    for (m, band), g in score.groupby(["model", "band"]):
        best = g.sort_values(["mean_rank", "val_loss"]).iloc[0]
        chosen.setdefault(m, {})[band] = float(best["lr"])
    return chosen, score


def merge_choice(previous, approach, chosen, grids):
    """Learning rates and grids to write: this run's models replace their entries,
    the others keep the ones in the previous file of the same approach."""
    if previous.get("approach") != approach:
        previous = {}
    old_rates = previous.get("learning_rates", {})
    old_grids = previous.get("lrs", {})
    if not isinstance(old_grids, dict):  # older files kept one grid for every model
        old_grids = {m: old_grids for m in old_rates}
    return {**old_rates, **chosen}, {**old_grids, **{m: g for m, g in grids.items() if m in chosen}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", default="finetuning", choices=list(APPROACHES))
    ap.add_argument("--models", nargs="+", default=None,
                    help="Default: every model of the approach (FT: bert roberta deberta; "
                         "IT: qwen_9b phi_mini gemma_12b flan-t5_base mistral_base zephyr_base).")
    ap.add_argument("--datasets", nargs="+", default=["sst2", "boolq", "ag_news", "snips"],
                    choices=list(DATASET_CONFIGS))
    ap.add_argument("--lrs", nargs="+", type=float, default=None,
                    help="One grid for every model. Default: FT 1e-5 2e-5 3e-5 5e-5; IT LoRA "
                         "2e-5 5e-5 1e-4 2e-4 5e-4, Flan-T5 1e-5 3e-5 1e-4 3e-4 1e-3.")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--select-only", action="store_true",
                    help="Only aggregate the finished runs and write the choice.")
    args = ap.parse_args()
    defaults = APPROACHES[args.approach]
    for key in ("models", "results_root", "logs_dir", "out"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    unknown = sorted(set(args.models) - set(defaults["models"]))
    if unknown:
        ap.error(f"--models {unknown} not in {args.approach}: {defaults['models']}")
    grids = ({m: args.lrs for m in args.models} if args.lrs
             else default_lrs(args.approach, args.models))

    exps = sweep_experiments(args.models, args.datasets, grids, args.results_root, args.approach)
    pending = [e for e in exps if not is_already_completed(e)]
    print(f"Sweep: {len(exps)} configurations, {len(pending)} pending")

    if args.dry_run:
        for e in pending:
            print(f"  {e['lr_key']} {e['band']} {e['dataset']} N={e['num_labelled']} lr={e['lr']:g}: "
                  + " ".join(e["cmd_args"]))
        return
    if not args.select_only:
        for i, e in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}]", end="")
            run_job({"members": [e], "cmd_args": e["cmd_args"],
                     "tag": f"lr_sweep_{e['lr_key']}_{e['dataset']}_N{e['num_labelled']}_lr{e['lr']:g}"},
                    gpu_id=args.gpus, logs_dir=args.logs_dir)

    table = load_sweep_results(exps)
    if table.empty:
        sys.exit("no sweep results found")
    missing = [e for e in exps if not is_already_completed(e)]
    if missing:
        print(f"WARNING: {len(missing)} incomplete configurations; the choice uses only finished ones.")
    chosen, score = select(table)

    os.makedirs(args.results_root, exist_ok=True)
    table.to_csv(os.path.join(args.results_root, "lr_sweep_table.csv"), index=False)
    score.to_csv(os.path.join(args.results_root, "lr_sweep_score.csv"), index=False)
    def show(df):
        print(df.round(4).assign(lr=df["lr"].map("{:g}".format)).to_string(index=False))
    show(table.sort_values(["model", "band", "dataset", "lr"]))
    show(score)

    previous = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            previous = json.load(f)
    learning_rates, all_grids = merge_choice(previous, args.approach, chosen, grids)
    with open(args.out, "w") as f:
        json.dump({
            "approach": args.approach,
            "criterion": defaults["criterion"],
            "bands": [{"name": b, "max_n": mx, "probe_n": pn} for b, mx, pn in FT_LR_BANDS],
            "datasets": args.datasets,
            "lrs": all_grids,
            "created": datetime.now().isoformat(timespec="seconds"),
            "learning_rates": learning_rates,
        }, f, indent=2)
    print(f"\nChosen LR per model and band -> {args.out}")
    for m, bands in learning_rates.items():
        kept = "" if m in chosen else "  (kept from the previous file)"
        print(f"  {m}: " + ", ".join(f"{b}={lr:g}" for b, lr in bands.items()) + kept)


if __name__ == "__main__":
    main()
