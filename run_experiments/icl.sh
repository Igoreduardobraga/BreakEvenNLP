#!/usr/bin/env bash
set -euo pipefail

DATASETS=("boolq" "ag_news" "snips" "sst2")
MODELS=("llama2" "flan-t5" "zephyr" "mistral")

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

  for M in "${MODELS[@]}"; do
    for N in "${SIZES[@]}"; do
      echo -e "\n======= ICL ${M} | dataset=${DS} | num_labelled=${N} =======\n"
        python3 main.py \
        --experiment_name icl \
        --configuration_name num_samples_${N} \
        --experiment_type icl \
        --dataset ${DS} \
        --model ${M} \
        --num_labelled ${N} \
        --full_test 0 \
        --num_labelled_test 1000 \
        --num_shots 2 \
        --batch_size 2 \
        --factor golden_model \
        --rskf_splits ${RSKFSPLITS} \
        --rskf_repeats 1 \
        --rskf_seed 27
    done
  done
done
