#!/usr/bin/env python3
"""
run_experiments.py
Orquestrador Python de experimentos para o BreakEvenNLP.
Reproduz com 100% de fidelidade todas as execuções e hiperparâmetros
definidos nos scripts originais de 'run_experiments/', com suporte a:
- Seleção granular por script, dataset ou modelo
- Dry-run para visualização prévia dos comandos
- Gestão de logs em arquivo por execução
- Retomada inteligente (pula execuções já finalizadas)
- Suporte a alocação de GPU (CUDA_VISIBLE_DEVICES)
"""

import os
import sys
import argparse
import subprocess
import time
from typing import List, Dict, Any, Optional

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


def generate_finetuning_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/finetuning.sh"""
    commands = []
    datasets = ["sst2", "boolq", "ag_news", "snips"]
    models = ["bert", "roberta"]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for m in models:
            for n in cfg["sizes"]:
                batch_size = get_finetuning_batch_size(n)
                cmd_args = [
                    sys.executable, "main.py",
                    "--experiment_name", "finetuning",
                    "--configuration_name", f"num_samples_{n}",
                    "--experiment_type", "finetuning",
                    "--dataset", ds,
                    "--model", m,
                    "--num_labelled", str(n),
                    "--full_test", "1",
                    "--batch_size", str(batch_size),
                    "--num_epochs", "10",
                    "--lr", "1e-5",
                    "--max_len", "128",
                    "--factor", "golden_model",
                    "--rskf_splits", str(cfg["rskf_splits"]),
                    "--rskf_repeats", "3",
                    "--rskf_seed", "27"
                ]
                commands.append({
                    "script_origin": "finetuning.sh",
                    "experiment_type": "finetuning",
                    "experiment_name": "finetuning",
                    "dataset": ds,
                    "model": m,
                    "model_size": "base",
                    "num_labelled": n,
                    "configuration_name": f"num_samples_{n}",
                    "rskf_splits": cfg["rskf_splits"],
                    "rskf_repeats": 3,
                    "cmd_args": cmd_args,
                })
    return commands


def generate_icl_commands() -> List[Dict[str, Any]]:
    """Gera comandos exatamente como definidos em run_experiments/icl.sh"""
    commands = []
    datasets = ["boolq", "ag_news", "snips", "sst2"]
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
                    "--full_test", "0",
                    "--num_labelled_test", "1000",
                    "--num_shots", "2",
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
    datasets = ["sst2", "boolq", "ag_news", "snips"]
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
                "--full_test", "0",
                "--num_labelled_test", "1000",
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
    datasets = ["sst2", "boolq", "ag_news", "snips"]
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
                "--full_test", "1",
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
    datasets = ["sst2", "boolq", "ag_news", "snips"]
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
                    "--num_labelled_test", "2000",
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
    datasets = ["ag_news", "snips", "boolq", "sst2"]
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
                "--full_test", "0",
                "--num_labelled", "1000",
                "--num_labelled_test", "1000",
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
    datasets = ["ag_news", "snips", "boolq", "sst2"]
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
            "--full_test", "0",
            "--num_labelled", "1000",
            "--num_labelled_test", "1000",
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


SCRIPTS_MAP = {
    "finetuning": generate_finetuning_commands,
    "icl": generate_icl_commands,
    "icl-llama3": generate_icl_llama3_commands,
    "it-flan-t5": generate_it_flan_t5_commands,
    "it-mistral-zephyr": generate_it_mistral_zephyr_commands,
    "prompting": generate_prompting_commands,
    "prompting-llama3": generate_prompting_llama3_commands,
}


def is_already_completed(exp: Dict[str, Any]) -> bool:
    """Verifica se todos os folds do experimento já possuem results.json gravado."""
    exp_name = exp["experiment_name"]
    exp_type = exp["experiment_type"]
    model = exp["model"]
    model_size = exp["model_size"]
    cfg_name = exp["configuration_name"]
    dataset = exp["dataset"]
    factor = "golden_model"

    target_dir = os.path.join("results", exp_name, f"{exp_type}_{model}_{model_size}", cfg_name, dataset, factor)
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


def run_experiment(exp: Dict[str, Any], gpu_id: Optional[str] = None, logs_dir: str = "logs") -> bool:
    """Executa um experimento com isolamento de GPU e redirecionamento de logs."""
    os.makedirs(logs_dir, exist_ok=True)
    exp_tag = f"{exp['experiment_type']}_{exp['model']}_{exp['dataset']}_N{exp['num_labelled']}"
    log_file = os.path.join(logs_dir, f"{exp_tag}.log")

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd_str = " ".join(exp["cmd_args"])
    print(f"\n[RUNNING] {exp['script_origin']} | {exp['model']} | {exp['dataset']} | N={exp['num_labelled']}")
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
            exp["cmd_args"],
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
        description="Orquestrador Python de experimentos para o BreakEvenNLP (100% fiel aos comandos originais)."
    )
    parser.add_argument(
        "--script",
        default="all",
        choices=["all", "finetuning", "icl", "icl-llama3", "it-flan-t5", "it-mistral-zephyr", "prompting", "prompting-llama3"],
        help="Qual script de experimento executar (default: all)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["sst2", "boolq", "ag_news", "snips"],
        help="Filtrar por datasets específicos."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filtrar por modelos específicos (ex: bert roberta llama3 flan-t5 mistral zephyr llama2)."
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
        help="Força a execução mesmo que o resultado (results.json) já exista."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        type=str,
        help="Diretório onde os logs individuais serão salvos (default: logs/)."
    )

    args = parser.parse_args()

    # 1. Coleta os comandos originais
    all_experiments: List[Dict[str, Any]] = []
    if args.script == "all":
        for script_name, generator in SCRIPTS_MAP.items():
            all_experiments.extend(generator())
    else:
        all_experiments.extend(SCRIPTS_MAP[args.script]())

    # 2. Aplica filtros se especificados
    filtered_experiments = []
    for exp in all_experiments:
        if args.datasets and exp["dataset"] not in args.datasets:
            continue
        if args.models and exp["model"] not in args.models:
            continue
        filtered_experiments.append(exp)

    total_found = len(filtered_experiments)
    print(f"\n=======================================================")
    print(f" BreakEvenNLP — Orquestrador de Experimentos")
    print(f" Total de configurações encontradas: {total_found}")
    print(f"=======================================================")

    if total_found == 0:
        print("Nenhum experimento correspondeu aos filtros fornecidos.")
        return

    # 3. Dry-run ou Execução
    if args.dry_run:
        print("\n[MODO DRY-RUN] Nenhum comando será executado. Listando comandos:\n")
        for idx, exp in enumerate(filtered_experiments, 1):
            already_done = is_already_completed(exp)
            status_tag = "[JÁ CONCLUÍDO]" if already_done else "[PENDENTE]"
            cmd_line = " ".join(exp["cmd_args"])
            print(f"{idx:3d}. {status_tag} ({exp['script_origin']}) {exp['model']} | {exp['dataset']} | N={exp['num_labelled']}")
            print(f"     Comando: {cmd_line}\n")
        print(f"Total: {total_found} comandos mapeados com 100% de fidelidade aos scripts originais.")
        return

    # 4. Execução real
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, exp in enumerate(filtered_experiments, 1):
        print(f"\n-------------------------------------------------------")
        print(f"Progresso: [{idx}/{total_found}]")

        if not args.force and is_already_completed(exp):
            print(f"[PULADO] Resultados já existem para {exp['model']} | {exp['dataset']} | N={exp['num_labelled']}. (Use --force para reexecutar)")
            skipped_count += 1
            continue

        ok = run_experiment(exp, gpu_id=args.gpus, logs_dir=args.logs_dir)
        if ok:
            success_count += 1
        else:
            failed_count += 1

    print(f"\n=======================================================")
    print(f" Execuções Concluídas!")
    print(f" Sucesso: {success_count} | Pulados: {skipped_count} | Falhas: {failed_count}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
