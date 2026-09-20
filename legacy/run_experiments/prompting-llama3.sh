#!/usr/bin/env bash
set -euo pipefail

DATASETS=("ag_news" "snips" "boolq" "sst2" )

for DS in "${DATASETS[@]}"; do
  case "$DS" in
    "sst2")
      SIZES=(10 50 100 250 500 1000 2500 5000 10000 15000 20000)
      RSKFSPLITS=5
      ;;
    "boolq")
      RSKFSPLITS=10
      ;;
    "ag_news")
      RSKFSPLITS=5
      ;;
    "snips")
      RSKFSPLITS=10
      ;;
    *)
      echo "Dataset '${DS}' não reconhecido, pulando..."
      continue
      ;;
  esac

    echo -e "\n======= PROMPT llama3 | dataset=${DS} | num_labelled (Single Run) =======\n"
    python3 main.py \
    --experiment_name prompting \
    --configuration_name num_samples_1000 \
    --experiment_type prompting \
    --factor golden_model \
    --dataset ${DS} \
    --model llama3 \
    --model_size 8b \
    --full_test 0 \
    --num_labelled 1000 \
    --num_labelled_test 1000 \
    --batch_size 8 \
    --prompt_format 0 \
    --rskf_splits ${RSKFSPLITS} \
    --rskf_repeats 1 \
    --rskf_seed 27
done
