#!/usr/bin/env bash
set -euo pipefail

DATASETS=("sst2" "boolq" "ag_news" "snips")

for DS in "${DATASETS[@]}"; do
  case "$DS" in
    "sst2")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000 15000 20000)
      ;;
    "boolq" | "ag_news")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000)
      ;;
    "snips")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000 15000)
      ;;
    *)
      echo "Dataset '${DS}' não reconhecido, pulando..."
      continue
      ;;
  esac

  for N in "${SIZES[@]}"; do
    echo -e "\n===> FT BERT | dataset=${DS} | num_labelled=${N}"
    python main.py \
      --experiment_name ft_bert_base_k_fold \
      --configuration_name ${DS}_num_${N} \
      --experiment_type finetuning \
      --dataset ${DS} \
      --model bert \
      --num_labelled ${N} \
      --full_test 1 \
      --batch_size 8 \
      --num_epochs 5 \
      --lr 1e-5 \
      --max_len 128 \
      --factor golden_model \
      --investigation_runs 1 \
      --mitigation_runs 10 \
      --k_folds 10
  done
done
