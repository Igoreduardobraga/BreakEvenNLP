#!/usr/bin/env bash
set -euo pipefail

DATASETS=("sst2" "boolq" "ag_news" "snips")
MODELS=("mistral" "zephyr")

for DS in "${DATASETS[@]}"; do
  case "$DS" in
    "sst2")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000 15000 20000)
      RSKFSPLITS=5
      ;;
    "boolq")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000)
      RSKFSPLITS=10
      ;;
    "ag_news")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000)
      RSKFSPLITS=5
      ;;
    "snips")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000 15000)
      RSKFSPLITS=10
      ;;
    *)
      echo "Dataset '${DS}' não reconhecido, pulando..."
      continue
      ;;
  esac

  for N in "${SIZES[@]}"; do
    for M in "${MODELS[@]}"; do
      echo -e "\n======= IT ${M} | dataset=${DS} | num_labelled=${N} =======\n"
        python3 main.py \
        --experiment_name instruction_tuning \
        --configuration_name num_samples_${N} \
        --experiment_type instruction_tuning \
        --dataset ${DS} \
        --model ${M} \
        --num_labelled ${N} \
        --full_test 1 \
        --batch_size 4 \
        --num_epochs 5 \
        --lr 1e-5 \
        --max_len 128 \
        --factor golden_model \
        --rskf_splits ${RSKFSPLITS} \
        --rskf_repeats 1 \
        --rskf_seed 27
    done
  done
done
