# BreakEvenNLP

How many labelled samples does fine-tuning a small model (BERT, RoBERTa, DeBERTa) need to match a large language model used through prompting, in-context learning (ICL) or instruction tuning (IT)? This repository runs that comparison over a grid of datasets × models × N (number of labelled samples) and plots the break-even point where the fine-tuning curve crosses each LLM curve.

The code grew out of a study of randomness factors in learning with limited labelled data; the original experiments are described in [analysis/EffectsOfRandomnessFactors.md](analysis/EffectsOfRandomnessFactors.md) and [analysis/ImpactOfDatasetSize.md](analysis/ImpactOfDatasetSize.md) (historical: their command lines predate the current CLI).

## Setup

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`:

```bash
uv sync                # creates .venv with torch 2.4.1 (CUDA 12.4), transformers, peft, ...
uv run pytest -q       # unit tests (the vLLM parity test is skipped without a GPU)
```

vLLM is not a dependency of `.venv` because its torch pin conflicts with torch 2.4.1. Prompting/ICL with `--engine vllm`, and the newer models that need a recent transformers (Qwen3.5, Gemma 4), run from a separate environment `.venv-vllm` with vLLM installed. Activate it (so `ninja` is on `PATH`) and disable the FlashInfer sampler, whose JIT build fails with nvcc 12.4:

```bash
source .venv-vllm/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
```

Datasets are downloaded from the Hugging Face Hub on first use (`data.py`) and cached in `~/.cache/huggingface/datasets`; loading needs network access. Gated models (Llama, Mistral) need `huggingface-cli login`.

## Running experiments

`run_experiments.py` generates the full grid, skips configurations whose results already exist and writes one log per run to `logs/`:

```bash
# fine-tuning
uv run python run_experiments.py --script finetuning --datasets sst2 boolq ag_news snips --models bert roberta deberta

# prompting / ICL / IT with the modern models (from .venv-vllm)
python run_experiments.py --script prompting-modern --models qwen phi gemma --engine vllm --decoding guided
python run_experiments.py --script icl-modern --models qwen phi gemma --model-sizes 9b mini 26b --engine vllm --decoding guided
python run_experiments.py --script it-modern --models qwen phi gemma --decoding guided
```

`--decoding guided` restricts every answer to one of the class names: through vLLM's structured outputs for prompting and ICL, and on HF, where the instruction-tuned model is evaluated, by masking every token that would take the answer off the class names.

Useful flags: `--dry-run` lists the commands without running them, `--force` reruns finished configurations, `--gpus 0` sets `CUDA_VISIBLE_DEVICES`. Prompting/ICL configurations of the same model are grouped into a single `main.py --sweep` process so the model is loaded once; `--no-group` runs one process per configuration. See `python run_experiments.py --help` for the list of scripts.

A single configuration can be run directly with `main.py` (see `python main.py --help`). Each run evaluates K folds of `RepeatedStratifiedKFold` (`--rskf_splits`, `--rskf_repeats`, `--rskf_seed 27`) over the concatenated splits of the dataset. All approaches are evaluated on the same stratified 1000-sample test subset of each fold (`--full_test 0 --num_labelled_test 1000`).

Note that the labelled pool takes `ceil(N / classes)` samples per class, so the effective N can be slightly larger than `--num_labelled` (SNIPS, 7 classes: N=10 uses 14 samples). The plots use the effective N.

## Results and analysis

Each run writes to `results/<experiment_type>/<type>_<model>_<size>[_vllm][_guided]/<configuration>/<dataset>/<factor>/`:

- `summary.parquet`: one row per fold with F1, seeds, duration and energy (GPU from NVML, CPU from RAPL or a utilization × TDP estimate; `energy_method` records which);
- `predictions.parquet`: per-sample gold label, prediction, generated text and prompt;
- `repeat_<r>_fold_<k>/results.json`: legacy per-fold output.

```bash
uv run python analysis/plot_break_even.py --results-dir results     # curves + break-even points in analysis/figures/
uv run python analysis/recompute_ft_subset_f1.py                    # F1 of old fine-tuning runs on the shared test subset
```

## Code map

| File | Role |
|---|---|
| `main.py` | One experiment: model × dataset × N over the RSKF folds, or a `--sweep` of them |
| `run_experiments.py` | Grid orchestrator |
| `data.py`, `sample_pool.py` | Dataset loading, stratified label selection, ICL shots |
| `prompter.py` | Prompt templates per model family and dataset, IT training pairs |
| `evaluator.py` | HF, vLLM and API inference adapters and answer parsing |
| `rng.py` | Isolated seeds per randomness factor |
| `result_store.py` | Atomic parquet/JSON result writing |
| `sustainability.py` | Energy and CO₂ per fold |
