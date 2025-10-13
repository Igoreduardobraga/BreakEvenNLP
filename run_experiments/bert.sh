#!/usr/bin/env bash
set -euo pipefail

DATASETS=("sst2" "mrpc" "cola" "rte" "boolq" "trec" "ag_news" "db_pedia" "snips")
SIZES=(10 50 100 250 500 1000 2500 5000 10000 15000 20000)

for DS in "${DATASETS[@]}"; do
  for N in "${SIZES[@]}"; do
    echo "===> FT BERT | dataset=${DS} | num_labelled=${N}"
    python main.py \
      --experiment_name ft_bert_base_all_datasets \
      --configuration_name ${DS}_num_${N} \
      --experiment_type finetuning \
      --dataset ${DS} \
      --model bert \
      --num_labelled ${N} \
      --full_test 1 \
      --batch_size 16 \
      --num_epochs 10 \
      --lr 1e-5 \
      --max_len 128 \
      --factor golden_model \
      --investigation_runs 1 \
      --mitigation_runs 1 \
      --k_folds 10 \
      --k_seed 42
  done
done
