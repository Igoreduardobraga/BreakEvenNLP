#!/usr/bin/env python3
"""Varredura de learning rate do fine-tuning, uma LR por modelo e faixa de N.

Para cada modelo, faixa de N (FT_LR_BANDS em run_experiments.py), dataset e
LR, roda o repeat 0 do RSKF no N representativo da faixa. A LR escolhida e a
de menor val_loss (melhor epoch, split de validacao de 10% do treino), pelo
rank medio entre datasets. O F1 de teste e salvo so para consulta: escolher
por ele daria ao FT uma vantagem otimista sobre os LLMs.

Os resultados vao para --results-root (results_lr_sweep/), separados da grade
principal. A escolha e gravada em ft_learning_rates.json, que
run_experiments.py --script finetuning le.

Uso:
    uv run python run_lr_sweep.py [--models bert roberta deberta]
        [--datasets sst2 boolq ag_news snips] [--lrs 1e-5 2e-5 3e-5 5e-5]
        [--gpus 0] [--dry-run] [--select-only]
"""
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

from result_store import experiment_dir_name
from run_experiments import (DATASET_CONFIGS, FT_LR_BANDS, FT_LR_FILE,
                             finetuning_experiment, is_already_completed, run_job)

DEFAULT_LRS = [1e-5, 2e-5, 3e-5, 5e-5]


def sweep_experiments(models, datasets, lrs, results_root):
    exps = []
    for m in models:
        for band, _, probe_n in FT_LR_BANDS:
            for ds in datasets:
                for lr in lrs:
                    exp = finetuning_experiment(
                        ds, m, probe_n, lr, rskf_repeats=1,
                        configuration_name=f"lr_{lr:g}_num_samples_{probe_n}",
                        results_root=results_root)
                    exp["band"] = band
                    exps.append(exp)
    return exps


def load_sweep_results(exps):
    rows = []
    for exp in exps:
        path = os.path.join(
            exp["results_root"], exp["experiment_name"],
            experiment_dir_name("finetuning", exp["model"], "base"),
            exp["configuration_name"], exp["dataset"], "golden_model", "summary.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        for col in ("val_loss", "val_f1_macro", "f1_macro"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        rows.append(dict(model=exp["model"], band=exp["band"], n=exp["num_labelled"],
                         dataset=exp["dataset"], lr=exp["lr"], folds=len(df),
                         val_folds=int(df["val_loss"].notna().sum()),
                         val_loss=df["val_loss"].mean(), val_f1_macro=df["val_f1_macro"].mean(),
                         test_f1_macro=df["f1_macro"].mean()))
    return pd.DataFrame(rows)


def select(table):
    """Menor val_loss por dataset vira rank 1; vence a LR de menor rank medio."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["bert", "roberta", "deberta"])
    ap.add_argument("--datasets", nargs="+", default=["sst2", "boolq", "ag_news", "snips"],
                    choices=list(DATASET_CONFIGS))
    ap.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    ap.add_argument("--results-root", default="results_lr_sweep")
    ap.add_argument("--logs-dir", default=os.path.join("logs", "lr_sweep"))
    ap.add_argument("--out", default=FT_LR_FILE)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--select-only", action="store_true",
                    help="So agrega o que ja rodou e grava a escolha.")
    args = ap.parse_args()

    exps = sweep_experiments(args.models, args.datasets, args.lrs, args.results_root)
    pending = [e for e in exps if not is_already_completed(e)]
    print(f"Varredura: {len(exps)} configuracoes, {len(pending)} pendentes")

    if args.dry_run:
        for e in pending:
            print(f"  {e['model']} {e['band']} {e['dataset']} N={e['num_labelled']} lr={e['lr']:g}: "
                  + " ".join(e["cmd_args"]))
        return
    if not args.select_only:
        for i, e in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}]", end="")
            run_job({"members": [e], "cmd_args": e["cmd_args"],
                     "tag": f"lr_sweep_{e['model']}_{e['dataset']}_N{e['num_labelled']}_lr{e['lr']:g}"},
                    gpu_id=args.gpus, logs_dir=args.logs_dir)

    table = load_sweep_results(exps)
    if table.empty:
        sys.exit("nenhum resultado da varredura encontrado")
    missing = [e for e in exps if not is_already_completed(e)]
    if missing:
        print(f"AVISO: {len(missing)} configuracoes incompletas; a escolha usa so o que terminou.")
    chosen, score = select(table)

    os.makedirs(args.results_root, exist_ok=True)
    table.to_csv(os.path.join(args.results_root, "lr_sweep_table.csv"), index=False)
    score.to_csv(os.path.join(args.results_root, "lr_sweep_score.csv"), index=False)
    def show(df):
        print(df.round(4).assign(lr=df["lr"].map("{:g}".format)).to_string(index=False))
    show(table.sort_values(["model", "band", "dataset", "lr"]))
    show(score)

    with open(args.out, "w") as f:
        json.dump({
            "criterion": "min val_loss (best epoch), mean rank across datasets",
            "bands": [{"name": b, "max_n": mx, "probe_n": pn} for b, mx, pn in FT_LR_BANDS],
            "datasets": args.datasets,
            "lrs": args.lrs,
            "created": datetime.now().isoformat(timespec="seconds"),
            "learning_rates": chosen,
        }, f, indent=2)
    print(f"\nLR escolhida por modelo e faixa -> {args.out}")
    for m, bands in chosen.items():
        print(f"  {m}: " + ", ".join(f"{b}={lr:g}" for b, lr in bands.items()))


if __name__ == "__main__":
    main()
