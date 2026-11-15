#!/usr/bin/env python3
"""
run_experiments.py
Experiment orchestrator for BreakEvenNLP.
Reproduces the runs and hyperparameters of the original scripts in
'legacy/run_experiments/', except for the test set, now shared by every
approach (TEST_SET_ARGS), the shots of Llama 3 ICL and of IT evaluated with
ICL, now the same as ICL's (ICL_SHOTS_ARGS), and the learning rates of
fine-tuning and IT, chosen per model and N band by run_lr_sweep.py.
Features:
- Selection by script, dataset or model
- Dry run that lists the commands without running them
- One log file per run
- Resume: skips the runs whose folds are all saved
- --force reruns from scratch, moving the previous results to results_old/
- GPU selection (CUDA_VISIBLE_DEVICES)
- Prompting/ICL grouping: the pending configurations of one model run in a
  single main.py --sweep that loads the model once (disable with --no-group)
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
    """Fine-tuning grid with the LR chosen per model and N band."""
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
        print(f" WARNING: no LR chosen in {FT_LR_FILE} for {', '.join(sorted(missing))}; "
              f"using {DEFAULT_FT_LR:g} (run run_lr_sweep.py first).")
    return commands


def generate_icl_commands() -> List[Dict[str, Any]]:
    """Commands of run_experiments/icl.sh."""
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
    """Commands of run_experiments/icl-llama3.sh, with the shots of ICL_SHOTS_ARGS instead of 5."""
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
                *ICL_SHOTS_ARGS,
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


def generate_prompting_commands() -> List[Dict[str, Any]]:
    """Commands of run_experiments/prompting.sh."""
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
    """Commands of run_experiments/prompting-llama3.sh."""
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
    """ICL commands for the modern models: Qwen 3.5 9B, Phi-4-mini, Gemma 4 12B."""
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
    """Zero-shot prompting commands for the modern models: Qwen 3.5 9B, Phi-4-mini, Gemma 4 12B."""
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
IT_FLAN_T5_MODELS = [("flan-t5", "base")]
IT_MISTRAL_ZEPHYR_MODELS = [("mistral", "base"), ("zephyr", "base")]

# IT LR per model and N band, chosen by run_lr_sweep.py
IT_LR_FILE = "it_learning_rates.json"
DEFAULT_IT_LR = 2e-4
DEFAULT_IT_LRS = {"flan-t5_base": 1e-5}


def it_lr_key(m: str, m_size: str) -> str:
    return f"{m}_{m_size}"


def default_it_lr(key: str) -> float:
    return DEFAULT_IT_LRS.get(key, DEFAULT_IT_LR)


def it_experiment(ds: str, m: str, m_size: str, n: int, lr: float,
                  rskf_splits: Optional[int] = None,
                  configuration_name: Optional[str] = None,
                  results_root: str = "results",
                  holdout_val: bool = False,
                  val_only: bool = False,
                  script_origin: str = "it-modern (run_experiments.py)") -> Dict[str, Any]:
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
        "script_origin": script_origin,
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


def generate_it_commands(models: List[Tuple[str, str]], script_origin: str) -> List[Dict[str, Any]]:
    """Instruction-tuning grid of the given models, with the LR chosen per model and N band."""
    datasets = ["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"]
    chosen = load_learning_rates(IT_LR_FILE)
    commands = []
    missing = set()
    for ds in datasets:
        for n in DATASET_CONFIGS[ds]["sizes"]:
            for m, m_size in models:
                key = it_lr_key(m, m_size)
                lr = chosen.get(key, {}).get(ft_lr_band(n))
                if lr is None:
                    lr = default_it_lr(key)
                    missing.add(f"{key}/{ft_lr_band(n)} ({lr:g})")
                commands.append(it_experiment(ds, m, m_size, n, lr, script_origin=script_origin))
    if missing:
        print(f" WARNING: no LR chosen in {IT_LR_FILE} for {', '.join(sorted(missing))}; using the "
              f"default in parentheses (run run_lr_sweep.py --approach instruction_tuning first).")
    return commands


def generate_it_modern_commands() -> List[Dict[str, Any]]:
    """Instruction-tuning commands for the modern models: Qwen 3.5 9B, Phi-4-mini, Gemma 4 12B."""
    return generate_it_commands(IT_MODERN_MODELS, "it-modern (run_experiments.py)")


def generate_it_flan_t5_commands() -> List[Dict[str, Any]]:
    """Commands of run_experiments/it-flan-t5.sh, with the LR chosen per N band instead of 1e-5."""
    return generate_it_commands(IT_FLAN_T5_MODELS, "it-flan-t5.sh")


def generate_it_mistral_zephyr_commands() -> List[Dict[str, Any]]:
    """Commands of run_experiments/it-mistral-zephyr.sh, with the LR chosen per model and N band
    instead of 1e-5."""
    return generate_it_commands(IT_MISTRAL_ZEPHYR_MODELS, "it-mistral-zephyr.sh")


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
    """Directory where main.py writes the experiment's folds."""
    exp_dir = experiment_dir_name(exp["experiment_type"], exp["model"], exp["model_size"],
                                  exp.get("engine", "hf"), exp.get("decoding", "free"))
    return os.path.join(exp.get("results_root", "results"), exp["experiment_name"], exp_dir,
                        exp["configuration_name"], exp["dataset"], "golden_model")


def is_already_completed(exp: Dict[str, Any]) -> bool:
    """True when every fold of the experiment has its results.json."""
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
    """Source and destination of the results that --force moves away before rerunning.

    Each directory goes to <root>_old/force_<stamp>/, under the same relative path.
    Without the old folds in place main.py reruns all of them, and an interrupted
    run resumes without --force instead of mixing old and new folds.
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
    """One job per main.py process: one experiment, or a group run through --sweep."""
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
            f"GROUP of {len(members)} configurations, model loaded once: {configs}")


def job_command(job: Dict[str, Any], logs_dir: str) -> List[str]:
    if "sweep" not in job:
        return job["cmd_args"]
    sweep_path = os.path.join(logs_dir, "sweeps", f"{job['tag']}.json")
    return job["base_args"] + ["--sweep", sweep_path]


def run_job(job: Dict[str, Any], gpu_id: Optional[str] = None, logs_dir: str = "logs") -> bool:
    """Runs a job on the selected GPU, writing its output to a log file."""
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
    print(f"          Command: {cmd_str}")

    start_time = time.time()
    with open(log_file, "a") as f_out:
        f_out.write(f"\n--- Started at: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f_out.write(f"Command: {cmd_str}\n\n")
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
        print(f"[SUCCESS] Finished in {elapsed:.1f}s")
        return True
    else:
        print(f"[FAILED] Exit code {ret_code} (see {log_file})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Experiment orchestrator for BreakEvenNLP."
    )
    parser.add_argument(
        "--script",
        default="all",
        choices=[
            "all", "finetuning", "icl", "icl-llama3", "icl-modern",
            "it-flan-t5", "it-mistral-zephyr", "it-modern",
            "prompting", "prompting-llama3", "prompting-modern"
        ],
        help="Experiment script to run (default: all)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["sst2", "boolq", "ag_news", "snips", "mrpc", "cola", "trec", "db_pedia"],
        help="Run only these datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run only these models (e.g. bert roberta qwen phi gemma llama3 flan-t5 mistral zephyr llama2)."
    )
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        default=None,
        help="Run only these model sizes (e.g. base 8b 9b mini 12b)."
    )
    parser.add_argument(
        "--gpus",
        default=None,
        type=str,
        help="GPU ID to use (e.g. '0' or '1'); sets CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the commands that would run, without running them."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the selected configurations from scratch: before running, move their existing "
             "results to results_old/force_<timestamp>/, under the same relative path. If the run "
             "stops midway, run it again without --force to resume where it stopped."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        type=str,
        help="Directory for the per-run logs (default: logs/)."
    )
    parser.add_argument(
        "--engine",
        default="hf",
        choices=["hf", "vllm"],
        help="Inference engine passed to main.py for prompting/ICL (default: hf); IT always evaluates on HF."
    )
    parser.add_argument(
        "--decoding",
        default="free",
        choices=["free", "guided"],
        help="Decoding passed to main.py for prompting, ICL and IT; guided restricts the answer to the class names."
    )
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="One process per configuration, without grouping prompting/ICL through --sweep."
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
    print(f" BreakEvenNLP — Experiment Orchestrator")
    print(f" Configurations found: {total_found}")
    print(f"=======================================================")

    if total_found == 0:
        print("No experiment matches the given filters.")
        return

    pending, done = [], []
    for exp in filtered_experiments:
        (done if not args.force and is_already_completed(exp) else pending).append(exp)
    jobs = build_jobs(pending, group=not args.no_group)
    print(f" Already done: {len(done)} | Pending: {len(pending)} in {len(jobs)} process(es)")

    if args.dry_run:
        print("\n[DRY RUN] No command will be run.\n")
        for exp in done:
            print(f"  [DONE] ({exp['script_origin']}) {exp['model']} | {exp['dataset']} | N={exp['num_labelled']}")
        if args.force:
            for src, dst in results_to_set_aside(pending, "<timestamp>"):
                print(f"  [WILL MOVE] {src} -> {dst}")
        for idx, job in enumerate(jobs, 1):
            print(f"\n{idx:3d}. [PENDING] {describe_job(job)}")
            print(f"     Command: {' '.join(job_command(job, args.logs_dir))}")
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
        print(f"Progress: [{idx}/{len(jobs)}]")
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
    print(f" Runs finished!")
    print(f" Succeeded: {success_count} | Skipped: {len(done)} | Failed: {failed_count}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
