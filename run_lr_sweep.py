#!/usr/bin/env python3
"""Varredura de learning rate do fine-tuning ou do instruction tuning, uma LR
por modelo e faixa de N.

Para cada modelo, faixa de N (FT_LR_BANDS em run_experiments.py), dataset e
LR, roda o repeat 0 do RSKF no N representativo da faixa. A LR escolhida e a
de menor val_loss (10% do treino), pelo rank medio entre datasets. Nas faixas
com early stopping e o val_loss do melhor epoch; na faixa pequena, que na grade
treina sem early stopping, e o val_loss do modelo final num holdout de 10%. O F1 de teste e salvo so para consulta: escolher
por ele daria ao FT uma vantagem otimista sobre os LLMs.

--approach instruction_tuning aplica o mesmo criterio a LR do LoRA dos modelos
novos, com o val_loss da resposta (a loss do treino). Como cada passo custa
muito mais que no FT, roda 2 folds por configuracao em vez do repeat inteiro e
nao avalia no teste (--it_val_only): o F1 de teste nao entra na escolha, e as
geracoes do teste, com e sem ICL, custariam mais que o treino nas faixas
pequena e media.

Os resultados vao para --results-root (results_lr_sweep/ no FT,
results_it_lr_sweep/ no IT), separados da grade principal. A escolha e gravada
em ft_learning_rates.json ou it_learning_rates.json, que run_experiments.py le
nos scripts finetuning e it-modern.

Uso:
    uv run python run_lr_sweep.py [--approach finetuning|instruction_tuning]
        [--models bert roberta deberta | qwen_9b phi_mini ...]
        [--datasets sst2 boolq ag_news snips] [--lrs ...]
        [--gpus 0] [--dry-run] [--select-only]
"""
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

from run_experiments import (DATASET_CONFIGS, FT_LR_BANDS, FT_LR_FILE, IT_LR_FILE,
                             IT_MODERN_MODELS, finetuning_experiment, is_already_completed,
                             it_lr_key, it_modern_experiment, results_dir, run_job)

IT_MODELS = {it_lr_key(m, size): (m, size) for m, size in IT_MODERN_MODELS}

# Por abordagem: modelos, LRs, raiz dos resultados, logs e arquivo da escolha.
# O LoRA varre uma ordem de grandeza acima do full fine-tuning.
APPROACHES = {
    "finetuning": dict(
        models=["bert", "roberta", "deberta"], lrs=[1e-5, 2e-5, 3e-5, 5e-5],
        results_root="results_lr_sweep", logs_dir=os.path.join("logs", "lr_sweep"), out=FT_LR_FILE),
    "instruction_tuning": dict(
        models=list(IT_MODELS), lrs=[5e-5, 1e-4, 2e-4, 5e-4],
        results_root="results_it_lr_sweep", logs_dir=os.path.join("logs", "it_lr_sweep"), out=IT_LR_FILE),
}

# Folds por configuracao na varredura do IT: RSKF com 2 splits, repeat 0.
IT_SWEEP_SPLITS = 2


def sweep_experiments(models, datasets, lrs, results_root, approach="finetuning"):
    exps = []
    for m in models:
        for band, _, probe_n in FT_LR_BANDS:
            for ds in datasets:
                for lr in lrs:
                    cfg_name = f"lr_{lr:g}_num_samples_{probe_n}"
                    if approach == "finetuning":
                        exp = finetuning_experiment(
                            ds, m, probe_n, lr, rskf_repeats=1, configuration_name=cfg_name,
                            results_root=results_root, holdout_val=True)
                    else:
                        exp = it_modern_experiment(
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
    ap.add_argument("--approach", default="finetuning", choices=list(APPROACHES))
    ap.add_argument("--models", nargs="+", default=None,
                    help="Default: todos os da abordagem (FT: bert roberta deberta; "
                         "IT: qwen_4b qwen_9b phi_mini gemma_26b).")
    ap.add_argument("--datasets", nargs="+", default=["sst2", "boolq", "ag_news", "snips"],
                    choices=list(DATASET_CONFIGS))
    ap.add_argument("--lrs", nargs="+", type=float, default=None,
                    help="Default: FT 1e-5 2e-5 3e-5 5e-5; IT 5e-5 1e-4 2e-4 5e-4.")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--select-only", action="store_true",
                    help="So agrega o que ja rodou e grava a escolha.")
    args = ap.parse_args()
    defaults = APPROACHES[args.approach]
    for key in ("models", "lrs", "results_root", "logs_dir", "out"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    unknown = sorted(set(args.models) - set(defaults["models"]))
    if unknown:
        ap.error(f"--models {unknown} fora de {args.approach}: {defaults['models']}")

    exps = sweep_experiments(args.models, args.datasets, args.lrs, args.results_root, args.approach)
    pending = [e for e in exps if not is_already_completed(e)]
    print(f"Varredura: {len(exps)} configuracoes, {len(pending)} pendentes")

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
            "approach": args.approach,
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
