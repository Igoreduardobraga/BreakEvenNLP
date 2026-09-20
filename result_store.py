"""
ResultStore: Atomic columnar metrics persistence and experiment tracking for BreakEvenNLP.
"""

from __future__ import annotations
import csv
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

SUMMARY_COLUMNS = [
    "experiment_name",
    "experiment_type",
    "model",
    "dataset",
    "factor",
    "configuration",
    "rskf_repeat",
    "rskf_fold",
    "run_seed_used",
    "data_split_seed",
    "label_choice_seed",
    "sample_choice_seed",
    "sample_order_seed",
    "model_initialisation_seed",
    "model_randomness_seed",
    "f1_macro",
    "f1_prompting",
    "f1_macro_icl",
    "duration_seconds",
    "timestamp",
    "git_commit",
]


def _get_git_commit() -> str:
    """Best-effort capture of current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


class ResultStore:
    """Deep module for atomic experiment metric tracking and persistence."""

    def __init__(
        self,
        results_path: Union[str, Path],
        experiment_name: str,
        experiment_type: str,
        model_name: str,
        dataset: str,
        factor: str,
        configuration_name: str = "",
        storage_format: str = "auto",
        legacy_json: bool = True,
    ) -> None:
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.model_name = model_name
        self.dataset = dataset
        self.factor = factor
        self.configuration_name = configuration_name
        self.storage_format = storage_format.lower()
        self.legacy_json = legacy_json

        self._git_commit = _get_git_commit()

    @property
    def summary_file(self) -> Path:
        """Determines the active summary file path based on format preference."""
        if self.storage_format == "parquet":
            try:
                import pyarrow  # noqa: F401
                return self.results_path / "summary.parquet"
            except ImportError:
                return self.results_path / "summary.csv"
        elif self.storage_format == "csv":
            return self.results_path / "summary.csv"
        else:  # 'auto'
            try:
                import pyarrow  # noqa: F401
                return self.results_path / "summary.parquet"
            except ImportError:
                return self.results_path / "summary.csv"

    def has_fold(self, repeat: int, fold: int) -> bool:
        """Checks if a fold has already been recorded."""
        # 1. Check legacy json directory
        legacy_path = self.results_path / f"repeat_{repeat}_fold_{fold}" / "results.json"
        if legacy_path.exists():
            return True

        # 2. Check summary table
        if self.summary_file.exists():
            if self.summary_file.suffix == ".csv":
                with open(self.summary_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if int(r.get("rskf_repeat", -1)) == repeat and int(r.get("rskf_fold", -1)) == fold:
                            return True
            elif self.summary_file.suffix == ".parquet":
                try:
                    import pandas as pd
                    df = pd.read_parquet(self.summary_file, columns=["rskf_repeat", "rskf_fold"])
                    match = df[(df["rskf_repeat"] == repeat) & (df["rskf_fold"] == fold)]
                    return not match.empty
                except Exception:
                    pass

        return False

    def load_fold_details(self, repeat: int, fold: int) -> Dict[str, Any]:
        """Loads detailed predictions, ground-truth targets, and decoded texts for a fold."""
        legacy_path = self.results_path / f"repeat_{repeat}_fold_{fold}" / "results.json"
        if legacy_path.exists():
            with open(legacy_path, "r", encoding="utf-8") as f:
                return json.load(f)

        raise FileNotFoundError(f"No fold details found for repeat {repeat}, fold {fold} in {self.results_path}")

    def load_summary(
        self,
        filters: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Any:
        """Loads and returns the summarized metrics table, optionally filtered.

        If as_dataframe is True and pandas is installed, returns a pd.DataFrame.
        Otherwise returns a list of dictionaries with parsed numeric fields.
        """
        if not self.summary_file.exists():
            if as_dataframe:
                try:
                    import pandas as pd
                    return pd.DataFrame(columns=SUMMARY_COLUMNS)
                except ImportError:
                    pass
            return []

        # Check if pandas is available and requested
        if as_dataframe:
            try:
                import pandas as pd
                if self.summary_file.suffix == ".parquet":
                    df = pd.read_parquet(self.summary_file)
                else:
                    df = pd.read_csv(self.summary_file)

                if filters:
                    for col, val in filters.items():
                        if col in df.columns:
                            if isinstance(val, (list, tuple, set)):
                                df = df[df[col].isin(val)]
                            else:
                                df = df[df[col] == val]
                return df
            except ImportError:
                pass

        # Fallback to standard library list of dicts
        rows: List[Dict[str, Any]] = []
        if self.summary_file.suffix == ".csv":
            with open(self.summary_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    parsed_row = self._parse_summary_row(r)
                    if self._matches_filters(parsed_row, filters):
                        rows.append(parsed_row)
        elif self.summary_file.suffix == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(self.summary_file)
                if filters:
                    for col, val in filters.items():
                        if col in df.columns:
                            if isinstance(val, (list, tuple, set)):
                                df = df[df[col].isin(val)]
                            else:
                                df = df[df[col] == val]
                return df.to_dict(orient="records")
            except Exception:
                pass

        return rows

    def _parse_summary_row(self, raw_row: Dict[str, str]) -> Dict[str, Any]:
        """Converts raw string values from CSV into appropriate Python types."""
        parsed: Dict[str, Any] = {}
        int_cols = {
            "rskf_repeat", "rskf_fold", "run_seed_used",
            "data_split_seed", "label_choice_seed", "sample_choice_seed",
            "sample_order_seed", "model_initialisation_seed", "model_randomness_seed",
            "prompt_format",
        }
        float_cols = {"f1_macro", "f1_prompting", "f1_macro_icl", "duration_seconds"}

        for k, v in raw_row.items():
            if v == "" or v is None:
                parsed[k] = None
            elif k in int_cols:
                try:
                    parsed[k] = int(v)
                except ValueError:
                    parsed[k] = v
            elif k in float_cols:
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            else:
                parsed[k] = v
        return parsed

    def _matches_filters(self, row: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Checks if a row matches the given filter conditions."""
        if not filters:
            return True
        for k, v in filters.items():
            row_val = row.get(k)
            if isinstance(v, (list, tuple, set)):
                if row_val not in v:
                    return False
            else:
                if row_val != v:
                    return False
        return True

    def record_fold(
        self,
        repeat: int,
        fold: int,
        run_seed_used: int,
        randomness_factor_seeds: Dict[str, int],
        metrics: Dict[str, float],
        real: Union[List[Any], Dict[str, Any], Any],
        predicted: Union[List[Any], Dict[str, Any], Any],
        decodeds: Optional[List[str]] = None,
        duration_seconds: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Atomically records the outcome of a single fold into the summary table and legacy JSON."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        row: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "model": self.model_name,
            "dataset": self.dataset,
            "factor": self.factor,
            "configuration": self.configuration_name,
            "rskf_repeat": int(repeat),
            "rskf_fold": int(fold),
            "run_seed_used": int(run_seed_used),
            "data_split_seed": randomness_factor_seeds.get("data_split_seed", ""),
            "label_choice_seed": randomness_factor_seeds.get("label_choice_seed", ""),
            "sample_choice_seed": randomness_factor_seeds.get("sample_choice_seed", ""),
            "sample_order_seed": randomness_factor_seeds.get("sample_order_seed", ""),
            "model_initialisation_seed": randomness_factor_seeds.get("model_initialisation_seed", ""),
            "model_randomness_seed": randomness_factor_seeds.get("model_randomness_seed", ""),
            "f1_macro": metrics.get("f1_macro", ""),
            "f1_prompting": metrics.get("f1_prompting", ""),
            "f1_macro_icl": metrics.get("f1_macro_icl", ""),
            "duration_seconds": round(duration_seconds, 4) if duration_seconds is not None else "",
            "timestamp": ts,
            "git_commit": self._git_commit,
        }

        self._atomic_append_summary(row, repeat=int(repeat), fold=int(fold))

        if self.legacy_json:
            self._write_legacy_json(
                repeat=int(repeat),
                fold=int(fold),
                run_seed_used=int(run_seed_used),
                randomness_factor_seeds=randomness_factor_seeds,
                metrics=metrics,
                real=real,
                predicted=predicted,
                decodeds=decodeds,
            )

    def _write_legacy_json(
        self,
        repeat: int,
        fold: int,
        run_seed_used: int,
        randomness_factor_seeds: Dict[str, int],
        metrics: Dict[str, float],
        real: Union[List[Any], Dict[str, Any], Any],
        predicted: Union[List[Any], Dict[str, Any], Any],
        decodeds: Optional[List[str]] = None,
    ) -> None:
        fold_dir = self.results_path / f"repeat_{repeat}_fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        target_path = fold_dir / "results.json"

        if self.experiment_type in ["instruction_tuning", "instruction_tuning_steps"]:
            legacy_dict: Dict[str, Any] = {
                "f1_prompting": float(metrics.get("f1_prompting", 0.0)),
                "f1_macro_icl": float(metrics.get("f1_macro_icl", 0.0)),
                **randomness_factor_seeds,
            }
        else:
            legacy_dict = {
                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                **randomness_factor_seeds,
            }

        legacy_dict["real"] = real
        legacy_dict["predicted"] = predicted
        legacy_dict["base_model"] = self.model_name
        legacy_dict["rskf_repeat"] = int(repeat)
        legacy_dict["rskf_fold"] = int(fold)
        legacy_dict["run_seed_used"] = int(run_seed_used)
        if decodeds is not None:
            legacy_dict["decodeds"] = decodeds

        with tempfile.NamedTemporaryFile("w", dir=fold_dir, delete=False, encoding="utf-8") as tf:
            tmp_path = Path(tf.name)
            json.dump(legacy_dict, tf, indent=4)
        os.replace(tmp_path, target_path)

    def _atomic_append_summary(self, new_row: Dict[str, Any], repeat: int, fold: int) -> None:
        """Thread/process-safe atomic write of summary row via staging and atomic replacement."""
        target_file = self.summary_file

        # If writing CSV
        if target_file.suffix == ".csv":
            existing_rows: List[Dict[str, Any]] = []
            if target_file.exists():
                with open(target_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)

            # Update or append
            updated = False
            for i, r in enumerate(existing_rows):
                if int(r.get("rskf_repeat", -1)) == repeat and int(r.get("rskf_fold", -1)) == fold:
                    existing_rows[i] = {k: str(new_row.get(k, "")) for k in SUMMARY_COLUMNS}
                    updated = True
                    break

            if not updated:
                existing_rows.append({k: str(new_row.get(k, "")) for k in SUMMARY_COLUMNS})

            # Atomic staging write
            dir_path = target_file.parent
            with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, encoding="utf-8", newline="") as tf:
                tmp_path = Path(tf.name)
                writer = csv.DictWriter(tf, fieldnames=SUMMARY_COLUMNS)
                writer.writeheader()
                writer.writerows(existing_rows)

            # Atomic rename / replace
            os.replace(tmp_path, target_file)

        elif target_file.suffix == ".parquet":
            import pandas as pd
            existing_df = None
            if target_file.exists():
                existing_df = pd.read_parquet(target_file)

            new_df = pd.DataFrame([new_row])
            if existing_df is not None and not existing_df.empty:
                mask = ~((existing_df["rskf_repeat"] == repeat) & (existing_df["rskf_fold"] == fold))
                combined_df = pd.concat([existing_df[mask], new_df], ignore_index=True)
            else:
                combined_df = new_df

            dir_path = target_file.parent
            with tempfile.NamedTemporaryFile("wb", dir=dir_path, delete=False, suffix=".parquet") as tf:
                tmp_path = Path(tf.name)
            combined_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, target_file)
