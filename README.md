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

Instruction tuning of Qwen3.5 also wants flash-linear-attention in `.venv-vllm`: without it transformers runs the Gated DeltaNet layers on a pure PyTorch path, about 10% slower per training step. Install it without dependencies, so torch and vLLM stay as they are:

```bash
uv pip install --python .venv-vllm/bin/python --no-deps flash-linear-attention==0.6.0 fla-core==0.6.0
```

Datasets are downloaded from the Hugging Face Hub on first use (`data.py`) and cached in `~/.cache/huggingface/datasets`; loading needs network access. Gated models (Llama, Mistral) need `huggingface-cli login`.

## Running experiments

`run_experiments.py` generates the full grid, skips configurations whose results already exist and writes one log per run to `logs/`:

```bash
# fine-tuning
uv run python run_experiments.py --script finetuning --datasets sst2 boolq ag_news snips --models bert roberta deberta

# prompting / ICL / IT with the modern models (from .venv-vllm)
python run_experiments.py --script prompting-modern --datasets sst2 boolq ag_news snips --models qwen phi gemma --engine vllm --decoding guided
python run_experiments.py --script icl-modern --datasets sst2 boolq ag_news snips --models qwen phi gemma --engine vllm --decoding guided
python run_experiments.py --script it-modern --datasets sst2 boolq ag_news snips --models qwen phi gemma --decoding guided
```

`--decoding guided` restricts every answer to one of the class names: through vLLM's structured outputs for prompting and ICL, and on HF, where the instruction-tuned model is evaluated, by masking every token that would take the answer off the class names.

Useful flags: `--dry-run` lists the commands without running them, `--force` reruns the selected configurations from scratch after moving their existing results to `results_old/force_<timestamp>/` (if a forced run stops midway, run it again without `--force` to resume), `--gpus 0` sets `CUDA_VISIBLE_DEVICES`. Prompting/ICL configurations of the same model are grouped into a single `main.py --sweep` process so the model is loaded once; `--no-group` runs one process per configuration. See `python run_experiments.py --help` for the list of scripts.

A single configuration can be run directly with `main.py` (see `python main.py --help`). Each run evaluates K folds of `RepeatedStratifiedKFold` (`--rskf_splits`, `--rskf_repeats`, `--rskf_seed 27`) over the concatenated splits of the dataset. All approaches are evaluated on the same stratified 1000-sample test subset of each fold (`--full_test 0 --num_labelled_test 1000`).

Instruction tuning evaluates each fold's trained model twice: zero-shot, the format it is trained on (`f1_prompting`), and with the same ICL shots as the ICL approach (`f1_macro_icl`, 2 per class); the plots draw them as the `instruction_tuning` and `instruction_tuning_icl` lines. The trained model is evaluated on HF in batches of 16 prompts (`HF_EVAL_BATCH_SIZE` in `main.py`).

Instruction tuning trains 5 epochs or 1250 optimizer steps, whichever comes first, but never less than one epoch (`IT_STEP_BUDGET` in `main.py`), for every model, Flan-T5 included. Every N up to 1000 trains all 5 epochs, N = 2500 about 2, N = 5000 about 1 and larger N exactly one, so every labelled sample is seen. With a validation split, the adapter kept is the best of 5 evenly spaced evaluations, one per epoch when every epoch trains.

Fine-tuning and the LoRA of instruction tuning train with a learning rate chosen per model and N band (N ≤ 100, ≤ 1000, larger) by validation loss, never by test F1. `run_lr_sweep.py` writes the choice to `ft_learning_rates.json` or `it_learning_rates.json`, which the grid reads; without a choice, IT uses 2e-4, the usual QLoRA rate. The IT sweep scores 2 folds per configuration and skips the test generations, which would cost more than the training. Each run rewrites the JSON with the models it was given, so run it with all of them (the default):

```bash
uv run python run_lr_sweep.py --gpus 0                                   # fine-tuning
python run_lr_sweep.py --approach instruction_tuning --gpus 0            # IT (from .venv-vllm)
```

Note that the labelled pool takes `ceil(N / classes)` samples per class, so the effective N can be slightly larger than `--num_labelled` (SNIPS, 7 classes: N=10 uses 14 samples). The plots use the effective N.

## Results and analysis

Each run writes to `results/<experiment_type>/<type>_<model>_<size>[_vllm][_guided]/<configuration>/<dataset>/<factor>/`:

- `summary.parquet`: one row per fold with F1, seeds, duration, energy and CO₂. GPU energy comes from NVML's energy counter, or from sampled power draw (NVML or `nvidia-smi`) when the counter is unavailable; CPU energy from RAPL or a utilization × TDP estimate (`CPU_TDP_WATTS`, default 85 W). `energy_method` records which was used: fine-tuning runs in `.venv`, without pynvml, so its GPU energy is sampled with `nvidia-smi`. `co2_kg` is energy times a fixed carbon intensity, 0.475 kg/kWh (world average) unless `CARBON_INTENSITY_KG_PER_KWH` is set (Brazil ≈ 0.074). A fold's cost covers its training and evaluation; the vLLM engine of prompting and ICL is built once per process, outside the folds, so its load is not counted;
- `predictions.parquet`: per-sample gold label, prediction, generated text and prompt;
- `repeat_<r>_fold_<k>/results.json`: legacy per-fold output.

```bash
uv run python analysis/plot_break_even.py --results-dir results     # curves + break-even points in analysis/figures/
uv run python analysis/plot_sustainability.py --metric energy       # cost per fold and F1 x cost (energy|co2|time)
uv run python analysis/recompute_ft_subset_f1.py                    # F1 of old fine-tuning runs on the shared test subset
```

`plot_break_even.py` draws every model by default: colour = approach, marker and dash = model, with the best model of each approach as a thick solid line (star in the legend). It writes `break_even_<dataset>.png`, `break_even_combined.png`, `break_even_by_model[_<dataset>].png` (one column per LLM against the best fine-tuning) and `break_even_agg.csv`. The break-even is the first N where the best fine-tuning line rises above the best line of each other approach (crossings at N ≤ 10 are ignored). `--best-only` draws one line per approach; give it another `--out-dir` so the two modes do not overwrite each other's PNGs.

`plot_sustainability.py` writes `sustainability_cost_<metric>.png` (cost per fold against N, log-log), `sustainability_tradeoff_<metric>.png` (macro-F1 against cost per fold) and `sustainability_agg.csv`, and logs the whole-grid cost per approach; `--best-only` keeps the best model of each approach. Instruction tuning + ICL is left out of the costs, since it is the same fold as instruction tuning.

## Code map

| File | Role |
|---|---|
| `main.py` | One experiment: model × dataset × N over the RSKF folds, or a `--sweep` of them |
| `run_experiments.py` | Grid orchestrator |
| `run_lr_sweep.py` | Learning rate per model and N band for fine-tuning and IT |
| `data.py`, `sample_pool.py` | Dataset loading, stratified label selection, ICL shots |
| `prompter.py` | Prompt templates per model family and dataset, IT training pairs |
| `evaluator.py` | HF, vLLM and API inference adapters and answer parsing |
| `rng.py` | Isolated seeds per randomness factor |
| `result_store.py` | Atomic parquet/JSON result writing |
| `sustainability.py` | Energy and CO₂ per fold |
| `analysis/plot_break_even.py` | Learning curves and break-even points |
| `analysis/plot_sustainability.py` | Energy, CO₂ and time per approach |
| `analysis/recompute_ft_subset_f1.py` | F1 of old fine-tuning runs on the shared test subset |
| `analysis/check_boolq_truncation.py` | Paired check of the BoolQ (question, passage) encoding against the old truncated text |
