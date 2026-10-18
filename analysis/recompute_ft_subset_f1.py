#!/usr/bin/env python3
"""Recalcula o F1 do fine-tuning no conjunto de teste comum (subconjunto
estratificado de 1000 amostras por fold) a partir das predicoes ja salvas.

Rodadas de FT antigas foram avaliadas no fold de teste inteiro, enquanto
prompting/ICL usam o subconjunto escolhido pela seed label_choice
(TEST_SET_ARGS em run_experiments.py). Este script reconstroi esse
subconjunto e recalcula o F1 sem treinar nada.

Verificacoes (abortam se falharem):
- a ordem de `real` no results.json bate com o fold reconstruido pelo RSKF;
- o subconjunto reconstruido e identico ao avaliado pelo prompting, quando
  houver predictions.parquet de prompting nos diretorios informados.

Saida: CSV com uma linha por fold (f1_full, f1_subset e tamanhos).

Uso:
    .venv/bin/python analysis/recompute_ft_subset_f1.py \
        [--results-dirs results results_legacy] \
        [--out analysis/figures/ft_subset_f1.csv] [--num-test 1000]
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_text_and_targets  # noqa: E402
from run_experiments import DATASET_CONFIGS  # noqa: E402
from sample_pool import SamplePool  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("recompute_ft_subset_f1")

RSKF_SEED = 27
FOLD_RE = re.compile(r"repeat_(\d+)_fold_(\d+)")
N_RE = re.compile(r"num_samples_(\d+)")


@lru_cache(None)
def dataset(ds):
    texts, targets, classes = load_text_and_targets(ds, 0)
    return texts, np.array(targets), classes


@lru_cache(None)
def splits(ds, n_splits, n_repeats):
    # As particoes de um repeat nao dependem de n_repeats (mesma sequencia de
    # RNG), entao basta n_repeats >= repeat+1 para reproduzir qualquer fold.
    _, y, _ = dataset(ds)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RSKF_SEED)
    return list(rskf.split(np.zeros(len(y)), y))


@lru_cache(None)
def subset(ds, n_splits, n_repeats, split_idx, label_seed, num_test):
    """(posicoes do subconjunto dentro do fold inteiro, rotulos na ordem avaliada)."""
    texts, y, classes = dataset(ds)
    train_idx, test_idx = splits(ds, n_splits, n_repeats)[split_idx]
    pool = SamplePool(ds, (list(train_idx), list(test_idx)), num_labelled=0,
                      full_test=False, num_labelled_test=num_test, label_seed=label_seed,
                      texts=texts, targets=tuple(y.tolist()), classes=classes)
    pos = {g: i for i, g in enumerate(test_idx)}
    return np.array([pos[g] for g in pool.test_indices]), np.array(pool.test_targets)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true), zero_division=0)


def verify_against_prompting(results_dirs, num_test):
    """O subconjunto reconstruido tem de ser exatamente o que o prompting avaliou."""
    checked = 0
    for root in results_dirs:
        for p in glob.glob(f"{root}/prompting/*/*/*/golden_model/predictions.parquet"):
            ds = Path(p).parents[1].name
            summ = pd.read_parquet(Path(p).with_name("summary.parquet"))
            preds = pd.read_parquet(p)
            n_splits = summ["rskf_fold"].nunique()
            for _, row in summ.iterrows():
                fold = int(row["rskf_fold"])
                _, gold = subset(ds, n_splits, 1, fold, int(row["label_choice_seed"]), num_test)
                saved = (preds[preds.rskf_fold == fold].sort_values("sample_id")
                         ["golden_label"].astype(int).to_numpy())
                if not np.array_equal(gold, saved):
                    raise SystemExit(f"subconjunto diverge do prompting: {p} fold {fold}")
                checked += 1
    return checked


def recompute(results_dirs, num_test):
    rows = []
    for root in results_dirs:
        for ds_dir in sorted(glob.glob(f"{root}/finetuning/finetuning_*/num_samples_*/*/golden_model")):
            ds_dir = Path(ds_dir)
            ds = ds_dir.parent.name
            n_splits = DATASET_CONFIGS[ds]["rskf_splits"]
            files = sorted(ds_dir.glob("repeat_*_fold_*/results.json"))
            if not files:
                continue
            folds = [tuple(map(int, FOLD_RE.search(str(f)).groups())) for f in files]
            n_repeats = max(r for r, _ in folds) + 1
            _, y, _ = dataset(ds)
            for f, (rep, fold) in zip(files, folds):
                d = json.load(open(f))
                split_idx = rep * n_splits + fold
                _, test_idx = splits(ds, n_splits, n_repeats)[split_idx]
                real, pred = np.array(d["real"]), np.array(d["predicted"])
                if len(real) != len(test_idx):
                    log.warning("ignorando %s: %d predicoes, fold inteiro tem %d "
                                "(ja avaliado no subconjunto?)", f, len(real), len(test_idx))
                    continue
                if not np.array_equal(real, y[test_idx]):
                    raise SystemExit(f"ordem das predicoes diverge do fold: {f}")
                pos, gold = subset(ds, n_splits, n_repeats, split_idx, int(d["label_choice"]), num_test)
                assert np.array_equal(real[pos], gold)
                rows.append(dict(
                    source=root,
                    exp_dir=ds_dir.parents[2].name,
                    dataset=ds,
                    n=int(N_RE.search(str(ds_dir)).group(1)),
                    rskf_repeat=rep,
                    rskf_fold=fold,
                    label_choice_seed=int(d["label_choice"]),
                    f1_full=f1(real, pred),
                    f1_subset=f1(real[pos], pred[pos]),
                    n_test_full=len(real),
                    n_test_subset=len(pos),
                ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dirs", nargs="+", default=["results", "results_legacy"])
    ap.add_argument("--out", default="analysis/figures/ft_subset_f1.csv")
    ap.add_argument("--num-test", type=int, default=1000,
                    help="tamanho do subconjunto de teste (igual a --num_labelled_test)")
    args = ap.parse_args()

    dirs = [d for d in args.results_dirs if os.path.isdir(d)]
    log.info("verificacao vs prompting: %d folds identicos", verify_against_prompting(dirs, args.num_test))
    df = recompute(dirs, args.num_test)
    if df.empty:
        raise SystemExit("nenhum results.json de fine-tuning avaliado no fold inteiro")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    diff = (df.f1_subset - df.f1_full) * 100
    log.info("salvo %s (%d folds); diferenca subset-full: media %.2f pp, |max| %.2f pp",
             args.out, len(df), diff.mean(), diff.abs().max())


if __name__ == "__main__":
    sys.exit(main())
