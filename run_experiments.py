#!/usr/bin/env python3
"""
run_experiments.py
Orquestrador Python de experimentos para o BreakEvenNLP.
Reproduz as execuções e hiperparâmetros dos scripts originais de
'legacy/run_experiments/', exceto o conjunto de teste, que agora é comum
a todas as abordagens (TEST_SET_ARGS), e os shots com que o IT é avaliado
em ICL, agora iguais aos do ICL (ICL_SHOTS_ARGS). Suporta:
- Seleção granular por script, dataset ou modelo
- Dry-run para visualização prévia dos comandos
- Gestão de logs em arquivo por execução
- Retomada inteligente (pula execuções já finalizadas)
- --force refaz do zero, guardando os resultados anteriores em results_old/
- Suporte a alocação de GPU (CUDA_VISIBLE_DEVICES)
- Agrupamento de prompting/ICL: as configurações pendentes de um mesmo
  modelo rodam num único main.py --sweep, carregando o modelo uma vez
  (desligue com --no-group)
"""

import os
import sys
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from typing import List, Dict, Any, Optional, Tuple

from result_store import experiment_dir_name

# Same stratified 1000-sample test subset per fold for every approach.
TEST_SET_ARGS = ["--full_test", "0", "--num_labelled_test", "1000"]

# ICL shots per class, also used by IT + ICL so it differs from ICL only by the tuning.
ICL_SHOTS_ARGS = ["--num_shots", "2"]

DATASET_CONFIGS = {
    "sst2": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000],
        "rskf_splits": 5,
    },
    "boolq": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        "rskf_splits": 10,
    },
    "ag_news": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        "rskf_splits": 5,
    },
    "snips": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000],
        "rskf_splits": 10,
    },
    "mrpc": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 4000],
        "rskf_splits": 5,
    },
    "cola": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 5000, 8000],
        "rskf_splits": 5,
    },
    "trec": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 4000],
        "rskf_splits": 10,
    },
    "db_pedia": {
        "sizes": [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000],
        "rskf_splits": 5,
    },
}


def get_finetuning_batch_size(n: int) -> int:
    if n <= 100:
        return 4
    elif n <= 500:
        return 8
    elif n <= 2500:
        return 16
    else:
        return 32


# Fine-tuning LR per model and N band, chosen by run_lr_sweep.py.
FT_LR_FILE = "ft_learning_rates.json"
FT_LR_BANDS = [("small", 100, 100), ("medium", 1000, 500), ("large", None, 2500)]
DEFAULT_FT_LR = 1e-5


def ft_lr_band(n: int) -> str:
    for name, max_n, _ in FT_LR_BANDS:
        if max_n is None or n <= max_n:
            return name
    raise ValueError(n)


def load_learning_rates(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)["learning_rates"]


def finetuning_experiment(ds: str, m: str, n: int, lr: float, rskf_repeats: int = 3,
                          configuration_name: Optional[str] = None,
                          results_root: str = "results",
                          holdout_val: bool = False) -> Dict[str, Any]:
    cfg = DATASET_CONFIGS[ds]
    cfg_name = configuration_name or f"num_samples_{n}"
    cmd_args = [
        sys.executable, "main.py",
        "--experiment_name", "finetuning",
        "--configuration_name", cfg_name,
        "--experiment_type", "finetuning",
        "--dataset", ds,
        "--model", m,
        "--num_labelled", str(n),
        *TEST_SET_ARGS,
        "--batch_size", str(get_finetuning_batch_size(n)),
        "--num_epochs", "10",
        "--lr", f"{lr:g}",
        "--max_len", "128",
        "--factor", "golden_model",
        "--rskf_splits", str(cfg["rskf_splits"]),
        "--rskf_repeats", str(rskf_repeats),
        "--rskf_seed", "27",
        "--results_root", results_root,
    ]
    if holdout_val:
        cmd_args += ["--holdout_val", "1"]
    return {
        "script_origin": "finetuning.sh",
        "experiment_type": "finetuning",
        "experiment_name": "finetuning",
        "dataset": ds,
        "model": m,
        "model_size": "base",
        "num_labelled": n,
        "lr": lr,
        "configuration_name": cfg_name,
        "rskf_splits": cfg["rskf_splits"],
        "rskf_repeats": rskf_repeats,
        "results_root": results_root,
        "cmd_args": cmd_args,
    }


def generate_finetuning_commands() -> List[Dict[str, Any]]:
    """Grade de fine-tuning com a LR escolhida por modelo e faixa de N."""
    datasets = ["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"]
    models = ["bert", "roberta", "deberta"]
    chosen = load_learning_rates(FT_LR_FILE)
    commands = []
    missing = set()
    for ds in datasets:
        for m in models:
            for n in DATASET_CONFIGS[ds]["sizes"]:
                lr = chosen.get(m, {}).get(ft_lr_band(n))
                if lr is None:
                    missing.add(f"{m}/{ft_lr_band(n)}")
                    lr = DEFAULT_FT_LR
                commands.append(finetuning_experiment(ds, m, n, lr))
    if missing:
        print(f" AVISO: sem LR escolhida em {FT_LR_FILE} para {', '.join(sorted(missing))}; "
              f"usando {DEFAULT_FT_LR:g} (rode run_lr_sweep.py antes).")
    return commands


def generate_icl_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/icl.sh"""
    commands = []
    datasets = ["boolq", "ag_news", "snips", "sst2", "mrpc", "cola", "trec", "db_pedia"]
    models = ["llama2", "flan-t5", "zephyr", "mistral"]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for m in models:
            for n in cfg["sizes"]:
                cmd_args = [
                    sys.executable, "main.py",
                    "--experiment_name", "icl",
                    "--configuration_name", f"num_samples_{n}",
                    "--experiment_type", "icl",
                    "--dataset", ds,
                    "--model", m,
                    "--num_labelled", str(n),
                    *TEST_SET_ARGS,
                    *ICL_SHOTS_ARGS,
                    "--batch_size", "2",
                    "--factor", "golden_model",
                    "--rskf_splits", str(cfg["rskf_splits"]),
                    "--rskf_repeats", "1",
                    "--rskf_seed", "27"
                ]
                commands.append({
                    "script_origin": "icl.sh",
                    "experiment_type": "icl",
                    "experiment_name": "icl",
                    "dataset": ds,
                    "model": m,
                    "model_size": "base",
                    "num_labelled": n,
                    "configuration_name": f"num_samples_{n}",
                    "rskf_splits": cfg["rskf_splits"],
                    "rskf_repeats": 1,
                    "cmd_args": cmd_args,
                })
    return commands


def generate_icl_llama3_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/icl-llama3.sh"""
    commands = []
    datasets = ["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"]
    model = "llama3"

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for n in cfg["sizes"]:
            cmd_args = [
                sys.executable, "main.py",
                "--experiment_name", "icl",
                "--configuration_name", f"num_samples_{n}",
                "--experiment_type", "icl",
                "--dataset", ds,
                "--model", model,
                "--model_size", "8b",
                "--num_labelled", str(n),
                *TEST_SET_ARGS,
                "--num_shots", "5",
                "--batch_size", "2",
                "--factor", "golden_model",
                "--rskf_splits", str(cfg["rskf_splits"]),
                "--rskf_repeats", "1",
                "--rskf_seed", "27"
            ]
            commands.append({
                "script_origin": "icl-llama3.sh",
                "experiment_type": "icl",
                "experiment_name": "icl",
                "dataset": ds,
                "model": model,
                "model_size": "8b",
                "num_labelled": n,
                "configuration_name": f"num_samples_{n}",
                "rskf_splits": cfg["rskf_splits"],
                "rskf_repeats": 1,
                "cmd_args": cmd_args,
            })
    return commands


def generate_it_flan_t5_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/it-flan-t5.sh"""
    commands = []
    datasets = ["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"]
    model = "flan-t5"

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for n in cfg["sizes"]:
            cmd_args = [
                sys.executable, "main.py",
                "--experiment_name", "instruction_tuning",
                "--configuration_name", f"num_samples_{n}",
                "--experiment_type", "instruction_tuning",
                "--dataset", ds,
                "--model", model,
                "--num_labelled", str(n),
                *TEST_SET_ARGS,
                *ICL_SHOTS_ARGS,
                "--batch_size", "4",
                "--num_epochs", "5",
                "--lr", "1e-5",
                "--max_len", "128",
                "--factor", "golden_model",
                "--rskf_splits", str(cfg["rskf_splits"]),
                "--rskf_repeats", "1",
                "--rskf_seed", "27"
            ]
            commands.append({
                "script_origin": "it-flan-t5.sh",
                "experiment_type": "instruction_tuning",
                "experiment_name": "instruction_tuning",
                "dataset": ds,
                "model": model,
                "model_size": "base",
                "num_labelled": n,
                "configuration_name": f"num_samples_{n}",
                "rskf_splits": cfg["rskf_splits"],
                "rskf_repeats": 1,
                "cmd_args": cmd_args,
            })
    return commands


def generate_it_mistral_zephyr_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/it-mistral-zephyr.sh"""
    commands = []
    datasets = ["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"]
    models = ["mistral", "zephyr"]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for n in cfg["sizes"]:
            for m in models:
                cmd_args = [
                    sys.executable, "main.py",
                    "--experiment_name", "instruction_tuning",
                    "--configuration_name", f"num_samples_{n}",
                    "--experiment_type", "instruction_tuning",
                    "--dataset", ds,
                    "--model", m,
                    "--num_labelled", str(n),
                    *TEST_SET_ARGS,
                    *ICL_SHOTS_ARGS,
                    "--batch_size", "4",
                    "--num_epochs", "5",
                    "--lr", "1e-5",
                    "--max_len", "128",
                    "--factor", "golden_model",
                    "--rskf_splits", str(cfg["rskf_splits"]),
                    "--rskf_repeats", "1",
                    "--rskf_seed", "27"
                ]
                commands.append({
                    "script_origin": "it-mistral-zephyr.sh",
                    "experiment_type": "instruction_tuning",
                    "experiment_name": "instruction_tuning",
                    "dataset": ds,
                    "model": m,
                    "model_size": "base",
                    "num_labelled": n,
                    "configuration_name": f"num_samples_{n}",
                    "rskf_splits": cfg["rskf_splits"],
                    "rskf_repeats": 1,
                    "cmd_args": cmd_args,
                })
    return commands


def generate_prompting_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/prompting.sh"""
    commands = []
    datasets = ["ag_news", "snips", "boolq", "sst2", "mrpc", "cola", "trec", "db_pedia"]
    models = ["llama2", "mistral", "zephyr", "flan-t5"]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for m in models:
            cmd_args = [
                sys.executable, "main.py",
                "--experiment_name", "prompting",
                "--configuration_name", "num_samples_1000",
                "--experiment_type", "prompting",
                "--factor", "golden_model",
                "--dataset", ds,
                "--model", m,
                *TEST_SET_ARGS,
                "--num_labelled", "1000",

                "--batch_size", "8",
                "--prompt_format", "0",
                "--rskf_splits", str(cfg["rskf_splits"]),
                "--rskf_repeats", "1",
                "--rskf_seed", "27"
            ]
            commands.append({
                "script_origin": "prompting.sh",
                "experiment_type": "prompting",
                "experiment_name": "prompting",
                "dataset": ds,
                "model": m,
                "model_size": "base",
                "num_labelled": 1000,
                "configuration_name": "num_samples_1000",
                "rskf_splits": cfg["rskf_splits"],
                "rskf_repeats": 1,
                "cmd_args": cmd_args,
            })
    return commands


def generate_prompting_llama3_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/prompting-llama3.sh"""
    commands = []
    datasets = ["ag_news", "snips", "boolq", "sst2", "mrpc", "cola", "trec", "db_pedia"]
    model = "llama3"

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        cmd_args = [
            sys.executable, "main.py",
            "--experiment_name", "prompting",
            "--configuration_name", "num_samples_1000",
            "--experiment_type", "prompting",
            "--factor", "golden_model",
            "--dataset", ds,
            "--model", model,
            "--model_size", "8b",
            *TEST_SET_ARGS,
            "--num_labelled", "1000",

            "--batch_size", "8",
            "--prompt_format", "0",
            "--rskf_splits", str(cfg["rskf_splits"]),
            "--rskf_repeats", "1",
            "--rskf_seed", "27"
        ]
        commands.append({
            "script_origin": "prompting-llama3.sh",
            "experiment_type": "prompting",
            "experiment_name": "prompting",
            "dataset": ds,
            "model": model,
            "model_size": "8b",
            "num_labelled": 1000,
            "configuration_name": "num_samples_1000",
            "rskf_splits": cfg["rskf_splits"],
            "rskf_repeats": 1,
            "cmd_args": cmd_args,
        })
    return commands


def generate_icl_modern_commands() -> List[Dict[str, Any]]:
    """Gera comandos de ICL para os novos modelos: Qwen 3.5 9B, Phi-4-mini, Gemma 4 12B"""
    commands = []
    datasets = ["boolq", "ag_news", "snips", "sst2", "mrpc", "cola", "trec", "db_pedia"]
    models = [
        ("qwen", "9b"),
        ("phi", "mini"),
        ("gemma", "12b"),
    ]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for m, m_size in models:
            for n in cfg["sizes"]:
                cmd_args = [
                    sys.executable, "main.py",
                    "--experiment_name", "icl",
                    "--configuration_name", f"num_samples_{n}",
                    "--experiment_type", "icl",
                    "--dataset", ds,
                    "--model", m,
                    "--model_size", m_size,
                    "--num_labelled", str(n),
                    *TEST_SET_ARGS,
                    *ICL_SHOTS_ARGS,
                    "--batch_size", "2",
                    "--factor", "golden_model",
                    "--rskf_splits", str(cfg["rskf_splits"]),
                    "--rskf_repeats", "1",
                    "--rskf_seed", "27"
                ]
                commands.append({
                    "script_origin": "icl-modern (run_experiments.py)",
                    "experiment_type": "icl",
                    "experiment_name": "icl",
                    "dataset": ds,
                    "model": m,
                    "model_size": m_size,
                    "num_labelled": n,
                    "configuration_name": f"num_samples_{n}",
                    "rskf_splits": cfg["rskf_splits"],
                    "rskf_repeats": 1,
                    "cmd_args": cmd_args,
                })
    return commands


def generate_prompting_modern_commands() -> List[Dict[str, Any]]:
    """Gera comandos de prompting zero-shot para os novos modelos: Qwen 3.5 9B, Phi-4-mini, Gemma 4 12B"""
    commands = []
    datasets = ["ag_news", "snips", "boolq", "sst2", "mrpc", "cola", "trec", "db_pedia"]
    models = [
        ("qwen", "9b"),
        ("phi", "mini"),
        ("gemma", "12b"),
    ]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for m, m_size in models:
            cmd_args = [
                sys.executable, "main.py",
                "--experiment_name", "prompting",
                "--configuration_name", "num_samples_1000",
                "--experiment_type", "prompting",
                "--factor", "golden_model",
                "--dataset", ds,
                "--model", m,
                "--model_size", m_size,
                *TEST_SET_ARGS,
                "--num_labelled", "1000",

                "--batch_size", "8",
                "--prompt_format", "0",
                "--rskf_splits", str(cfg["rskf_splits"]),
                "--rskf_repeats", "1",
                "--rskf_seed", "27"
            ]
            commands.append({
                "script_origin": "prompting-modern (run_experiments.py)",
                "experiment_type": "prompting",
                "experiment_name": "prompting",
                "dataset": ds,
                "model": m,
                "model_size": m_size,
                "num_labelled": 1000,
                "configuration_name": "num_samples_1000",
                "rskf_splits": cfg["rskf_splits"],
                "rskf_repeats": 1,
                "cmd_args": cmd_args,
            })
    return commands


IT_MODERN_MODELS = [("qwen", "9b"), ("phi", "mini"), ("gemma", "12b")]

# IT LoRA LR per model and N band, chosen by run_lr_sweep.py
IT_LR_FILE = "it_learning_rates.json"
DEFAULT_IT_LR = 2e-4


def it_lr_key(m: str, m_size: str) -> str:
    return f"{m}_{m_size}"


def it_modern_experiment(ds: str, m: str, m_size: str, n: int, lr: float,
                         rskf_splits: Optional[int] = None,
                         configuration_name: Optional[str] = None,
                         results_root: str = "results",
                         holdout_val: bool = False,
                         val_only: bool = False) -> Dict[str, Any]:
    cfg = DATASET_CONFIGS[ds]
    cfg_name = configuration_name or f"num_samples_{n}"
    splits = rskf_splits or cfg["rskf_splits"]
    cmd_args = [
        sys.executable, "main.py",
        "--experiment_name", "instruction_tuning",
        "--configuration_name", cfg_name,
        "--experiment_type", "instruction_tuning",
        "--dataset", ds,
        "--model", m,
        "--model_size", m_size,
        "--num_labelled", str(n),
        *TEST_SET_ARGS,
        *ICL_SHOTS_ARGS,
        "--batch_size", "4",
        "--num_epochs", "5",
        "--lr", f"{lr:g}",
        "--max_len", "128",
        "--factor", "golden_model",
        "--rskf_splits", str(splits),
        "--rskf_repeats", "1",
        "--rskf_seed", "27",
        "--results_root", results_root,
    ]
    if holdout_val:
        cmd_args += ["--holdout_val", "1"]
    if val_only:
        cmd_args += ["--it_val_only", "1"]
    return {
        "script_origin": "it-modern (run_experiments.py)",
        "experiment_type": "instruction_tuning",
        "experiment_name": "instruction_tuning",
        "dataset": ds,
        "model": m,
        "model_size": m_size,
        "num_labelled": n,
        "lr": lr,
        "configuration_name": cfg_name,
        "rskf_splits": splits,
        "rskf_repeats": 1,
        "results_root": results_root,
        "cmd_args": cmd_args,
    }


def generate_it_modern_commands() -> List[Dict[str, Any]]:
    """Gera comandos de instruction tuning para os novos modelos, com a LR escolhida por modelo e faixa de N."""
    datasets = ["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"]
    chosen = load_learning_rates(IT_LR_FILE)
    commands = []
    missing = set()
    for ds in datasets:
        for n in DATASET_CONFIGS[ds]["sizes"]:
            for m, m_size in IT_MODERN_MODELS:
                key = it_lr_key(m, m_size)
                lr = chosen.get(key, {}).get(ft_lr_band(n))
                if lr is None:
                    missing.add(f"{key}/{ft_lr_band(n)}")
                    lr = DEFAULT_IT_LR
                commands.append(it_modern_experiment(ds, m, m_size, n, lr))
    if missing:
        print(f" AVISO: sem LR escolhida em {IT_LR_FILE} para {', '.join(sorted(missing))}; "
              f"usando {DEFAULT_IT_LR:g} (rode run_lr_sweep.py --approach instruction_tuning antes).")
    return commands


SCRIPTS_MAP = {
    "finetuning": generate_finetuning_commands,
    "icl": generate_icl_commands,
    "icl-llama3": generate_icl_llama3_commands,
    "icl-modern": generate_icl_modern_commands,
    "it-flan-t5": generate_it_flan_t5_commands,
    "it-mistral-zephyr": generate_it_mistral_zephyr_commands,
    "it-modern": generate_it_modern_commands,
    "prompting": generate_prompting_commands,
    "prompting-llama3": generate_prompting_llama3_commands,
    "prompting-modern": generate_prompting_modern_commands,
}


def results_dir(exp: Dict[str, Any]) -> str:
    """Pasta em que o main.py grava os folds do experimento."""
    exp_dir = experiment_dir_name(exp["experiment_type"], exp["model"], exp["model_size"],
                                  exp.get("engine", "hf"), exp.get("decoding", "free"))
    return os.path.join(exp.get("results_root", "results"), exp["experiment_name"], exp_dir,
                        exp["configuration_name"], exp["dataset"], "golden_model")


def is_already_completed(exp: Dict[str, Any]) -> bool:
    """Verifica se todos os folds do experimento já possuem results.json gravado."""
    target_dir = results_dir(exp)
    if not os.path.isdir(target_dir):
        return False

    total_expected = exp["rskf_repeats"] * exp["rskf_splits"]
    completed_count = 0
    for r in range(exp["rskf_repeats"]):
        for k in range(exp["rskf_splits"]):
            res_file = os.path.join(target_dir, f"repeat_{r}_fold_{k}", "results.json")
            if os.path.exists(res_file):
                completed_count += 1

    return completed_count == total_expected


def results_to_set_aside(experiments: List[Dict[str, Any]], stamp: str) -> List[Tuple[str, str]]:
    """Origem e destino dos resultados que o --force tira do lugar antes de refazer.

    Cada pasta vai para <raiz>_old/force_<stamp>/, no mesmo caminho relativo.
    Sem os folds antigos no lugar o main.py refaz todos, e uma execução
    interrompida continua sem --force em vez de misturar folds antigos e novos.
    """
    moves, seen = [], set()
    for exp in experiments:
        src = results_dir(exp)
        if src in seen or not os.path.isdir(src) or not os.listdir(src):
            continue
        seen.add(src)
        root = os.path.normpath(exp.get("results_root", "results"))
        moves.append((src, os.path.join(f"{root}_old", f"force_{stamp}", os.path.relpath(src, root))))
    return moves


GROUPABLE_TYPES = ("prompting", "icl", "icl_similarity")

# Flags that may differ within a group; all others must match.
SWEEP_FLAGS = {
    "--dataset": ("dataset", str),
    "--num_labelled": ("num_labelled", int),
    "--configuration_name": ("configuration_name", str),
    "--rskf_splits": ("rskf_splits", int),
}


def _strip_sweep_flags(cmd_args: List[str]) -> List[str]:
    out, skip = [], False
    for arg in cmd_args:
        if skip:
            skip = False
        elif arg in SWEEP_FLAGS:
            skip = True
        else:
            out.append(arg)
    return out


def _sweep_entry(cmd_args: List[str]) -> Dict[str, Any]:
    entry = {}
    for flag, (key, cast) in SWEEP_FLAGS.items():
        entry[key] = cast(cmd_args[cmd_args.index(flag) + 1])
    return entry


def _exp_tag(exp: Dict[str, Any]) -> str:
    return f"{exp['experiment_type']}_{exp['model']}_{exp['dataset']}_N{exp['num_labelled']}"


def build_jobs(experiments: List[Dict[str, Any]], group: bool) -> List[Dict[str, Any]]:
    """Um job por processo main.py: um experimento, ou um grupo via --sweep."""
    jobs: List[Dict[str, Any]] = []
    groups: Dict[tuple, Dict[str, Any]] = {}
    for exp in experiments:
        if not group or exp["experiment_type"] not in GROUPABLE_TYPES:
            jobs.append({"members": [exp], "cmd_args": exp["cmd_args"], "tag": _exp_tag(exp)})
            continue
        key = tuple(_strip_sweep_flags(exp["cmd_args"]))
        if key not in groups:
            groups[key] = {"members": [], "base_args": list(key)}
            jobs.append(groups[key])
        groups[key]["members"].append(exp)

    for job in jobs:
        if "base_args" not in job:
            continue
        members = job["members"]
        if len(members) == 1:
            job.update(cmd_args=members[0]["cmd_args"], tag=_exp_tag(members[0]))
            continue
        first = members[0]
        digest = hashlib.sha1(" ".join(job["base_args"]).encode()).hexdigest()[:8]
        job["tag"] = f"{first['experiment_type']}_{first['model']}_{first['model_size']}_sweep_{digest}"
        job["sweep"] = [_sweep_entry(m["cmd_args"]) for m in members]
    return jobs


def describe_job(job: Dict[str, Any]) -> str:
    members = job["members"]
    first = members[0]
    if "sweep" not in job:
        return f"({first['script_origin']}) {first['model']} | {first['dataset']} | N={first['num_labelled']}"
    configs = ", ".join(f"{m['dataset']}:N={m['num_labelled']}" for m in members)
    return (f"({first['script_origin']}) {first['model']} {first['model_size']} | "
            f"GRUPO com {len(members)} configurações, modelo carregado 1x: {configs}")


def job_command(job: Dict[str, Any], logs_dir: str) -> List[str]:
    if "sweep" not in job:
        return job["cmd_args"]
    sweep_path = os.path.join(logs_dir, "sweeps", f"{job['tag']}.json")
    return job["base_args"] + ["--sweep", sweep_path]


def run_job(job: Dict[str, Any], gpu_id: Optional[str] = None, logs_dir: str = "logs") -> bool:
    """Executa um job com isolamento de GPU e redirecionamento de logs."""
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{job['tag']}.log")
    cmd_args = job_command(job, logs_dir)
    if "sweep" in job:
        sweep_path = cmd_args[-1]
        os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(job["sweep"], f, indent=2)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd_str = " ".join(cmd_args)
    print(f"\n[RUNNING] {describe_job(job)}")
    if gpu_id is not None:
        print(f"          GPU: {gpu_id}")
    print(f"          Log: {log_file}")
    print(f"          Comando: {cmd_str}")

    start_time = time.time()
    with open(log_file, "a") as f_out:
        f_out.write(f"\n--- Iniciado em: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f_out.write(f"Comando: {cmd_str}\n\n")
        f_out.flush()

        process = subprocess.Popen(
            cmd_args,
            env=env,
            stdout=f_out,
            stderr=subprocess.STDOUT
        )
        ret_code = process.wait()

    elapsed = time.time() - start_time
    if ret_code == 0:
        print(f"[SUCCESS] Concluído em {elapsed:.1f}s")
        return True
    else:
        print(f"[FAILED] Erro com código {ret_code} (veja detalhes em {log_file})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador Python de experimentos para o BreakEvenNLP."
    )
    parser.add_argument(
        "--script",
        default="all",
        choices=[
            "all", "finetuning", "icl", "icl-llama3", "icl-modern",
            "it-flan-t5", "it-mistral-zephyr", "it-modern",
            "prompting", "prompting-llama3", "prompting-modern"
        ],
        help="Qual script de experimento executar (default: all)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"],
        help="Filtrar por datasets específicos."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filtrar por modelos específicos (ex: bert roberta llama3 flan-t5 mistral zephyr llama2)."
    )
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        default=None,
        help="Filtrar por tamanhos (ex: base 8b 4b 9b mini 12b)."
    )
    parser.add_argument(
        "--gpus",
        default=None,
        type=str,
        help="ID da GPU a utilizar (ex: '0' ou '1'). Define CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista todos os comandos que seriam executados, sem rodá-los."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refaz as configurações selecionadas do zero: antes de rodar, move os resultados que já "
             "existem para results_old/force_<data-hora>/, no mesmo caminho relativo. Se a execução "
             "parar no meio, rode de novo sem --force para continuar de onde parou."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        type=str,
        help="Diretório onde os logs individuais serão salvos (default: logs/)."
    )
    parser.add_argument(
        "--engine",
        default="hf",
        choices=["hf", "vllm"],
        help="Motor de inferência do prompting/ICL repassado ao main.py (default: hf); o IT sempre avalia no HF."
    )
    parser.add_argument(
        "--decoding",
        default="free",
        choices=["free", "guided"],
        help="Decoding repassado ao main.py no prompting, ICL e IT; guided restringe a resposta aos nomes das classes."
    )
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Um processo por configuração, sem agrupar prompting/ICL via --sweep."
    )

    args = parser.parse_args()

    all_experiments: List[Dict[str, Any]] = []
    if args.script == "all":
        for script_name, generator in SCRIPTS_MAP.items():
            all_experiments.extend(generator())
    else:
        all_experiments.extend(SCRIPTS_MAP[args.script]())

    filtered_experiments = []
    for exp in all_experiments:
        if args.datasets and exp["dataset"] not in args.datasets:
            continue
        if args.models and exp["model"] not in args.models:
            continue
        if args.model_sizes and exp.get("model_size") not in args.model_sizes:
            continue
        if exp["experiment_type"] in ("prompting", "icl", "icl_similarity"):
            exp["cmd_args"] = list(exp["cmd_args"]) + [
                "--engine", args.engine, "--decoding", args.decoding]
            exp["engine"] = args.engine
            exp["decoding"] = args.decoding
        elif exp["experiment_type"] in ("instruction_tuning", "instruction_tuning_steps"):
            exp["cmd_args"] = list(exp["cmd_args"]) + ["--decoding", args.decoding]
            exp["decoding"] = args.decoding
        filtered_experiments.append(exp)

    total_found = len(filtered_experiments)
    print(f"\n=======================================================")
    print(f" BreakEvenNLP — Orquestrador de Experimentos")
    print(f" Total de configurações encontradas: {total_found}")
    print(f"=======================================================")

    if total_found == 0:
        print("Nenhum experimento correspondeu aos filtros fornecidos.")
        return

    pending, done = [], []
    for exp in filtered_experiments:
        (done if not args.force and is_already_completed(exp) else pending).append(exp)
    jobs = build_jobs(pending, group=not args.no_group)
    print(f" Já concluídas: {len(done)} | Pendentes: {len(pending)} em {len(jobs)} processo(s)")

    if args.dry_run:
        print("\n[MODO DRY-RUN] Nenhum comando será executado.\n")
        for exp in done:
            print(f"  [JÁ CONCLUÍDO] ({exp['script_origin']}) {exp['model']} | {exp['dataset']} | N={exp['num_labelled']}")
        if args.force:
            for src, dst in results_to_set_aside(pending, "<data-hora>"):
                print(f"  [SERÁ MOVIDO] {src} -> {dst}")
        for idx, job in enumerate(jobs, 1):
            print(f"\n{idx:3d}. [PENDENTE] {describe_job(job)}")
            print(f"     Comando: {' '.join(job_command(job, args.logs_dir))}")
        return

    if args.force:
        # Move everything first, so an interrupted run resumes without --force.
        for src, dst in results_to_set_aside(pending, time.strftime("%Y%m%d-%H%M%S")):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f" [--force] {src} -> {dst}")

    success_count = 0
    failed_count = 0

    for idx, job in enumerate(jobs, 1):
        print(f"\n-------------------------------------------------------")
        print(f"Progresso: [{idx}/{len(jobs)}]")
        ok = run_job(job, gpu_id=args.gpus, logs_dir=args.logs_dir)
        if "sweep" in job:
            # A group may fail partway: count per configuration.
            finished = sum(is_already_completed(m) for m in job["members"])
            success_count += finished
            failed_count += len(job["members"]) - finished
        elif ok:
            success_count += 1
        else:
            failed_count += 1

    print(f"\n=======================================================")
    print(f" Execuções Concluídas!")
    print(f" Sucesso: {success_count} | Pulados: {len(done)} | Falhas: {failed_count}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
