"""
SustainabilityTracker: lightweight energy/carbon estimation per experiment fold.

Methodology (estimate, not meter-grade):
- Wall duration measured with time.monotonic().
- GPU power sampled via pynvml if installed, else `nvidia-smi` if present,
  else a conservative TDP fallback (250W per visible CUDA device when torch
  reports CUDA, 0 otherwise). Start/end samples are averaged.
- CPU power estimated as 15W base + 10W per logical core (psutil if available,
  else os.cpu_count). Documented constant, clearly an approximation.
- energy_kwh = (gpu_watts + cpu_watts) * duration_s / 3_600_000
- co2_kg = energy_kwh * carbon_intensity (default 0.475 kg/kWh world average,
  override via arg or CARBON_INTENSITY_KG_PER_KWH env for regional grids,
  e.g. Brazil ~0.074).

Usage in main.py (minimal diff, no re-indentation):
    tracker = SustainabilityTracker()
    tracker.start()
    ... run fold ...
    report = tracker.stop()
    result_store.record_fold(..., duration_seconds=report.duration_seconds,
                             energy_kwh=report.energy_kwh, co2_kg=report.co2_kg,
                             hardware=report.hardware,
                             carbon_intensity=report.carbon_intensity)

Also usable as context manager: `with SustainabilityTracker() as t: ...`.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

DEFAULT_CARBON_INTENSITY = 0.475
FALLBACK_GPU_TDP_WATTS = 250.0
CPU_BASE_WATTS = 15.0
CPU_PER_CORE_WATTS = 10.0


def _resolve_carbon_intensity(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return float(explicit)
    env = os.environ.get("CARBON_INTENSITY_KG_PER_KWH")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return DEFAULT_CARBON_INTENSITY


def _cpu_watts() -> float:
    try:
        import psutil
        cores = psutil.cpu_count(logical=True) or (os.cpu_count() or 1)
    except Exception:
        cores = os.cpu_count() or 1
    return CPU_BASE_WATTS + CPU_PER_CORE_WATTS * max(1, cores)


def _cuda_device_count() -> int:
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except Exception:
        pass
    return 0


def _hardware_name() -> str:
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                return str(torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception:
        pass
    return str(platform.machine() or platform.processor() or "cpu")


def _read_gpu_power_watts() -> Optional[float]:
    """Total GPU power draw in watts, or None if unmeasurable."""
    n = _cuda_device_count()
    if n == 0:
        return 0.0
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
            total_mw = 0
            for i in range(n):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                total_mw += pynvml.nvmlDeviceGetPowerUsage(handle)
            return total_mw / 1000.0
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except ImportError:
        pass
    if shutil.which("nvidia-smi") is not None:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5, check=False,
            )
            if res.returncode == 0:
                total = 0.0
                found = False
                for line in res.stdout.strip().splitlines():
                    try:
                        total += float(line.strip().split()[0])
                        found = True
                    except (ValueError, IndexError):
                        pass
                if found:
                    return total
        except Exception:
            pass
    return None


@dataclass
class SustainabilityReport:
    duration_seconds: float
    energy_kwh: float
    co2_kg: float
    hardware: str
    carbon_intensity: float
    gpu_count: int
    method: str


class SustainabilityTracker:
    """Measures wall time and estimates energy/carbon for one fold."""

    def __init__(self, carbon_intensity: Optional[float] = None):
        self.carbon_intensity = _resolve_carbon_intensity(carbon_intensity)
        self.hardware = _hardware_name()
        self.gpu_count = _cuda_device_count()
        self._start_monotonic: Optional[float] = None
        self._start_gpu_watts: Optional[float] = None
        self._report: Optional[SustainabilityReport] = None

    def start(self) -> "SustainabilityTracker":
        self._report = None
        self._start_monotonic = time.monotonic()
        self._start_gpu_watts = _read_gpu_power_watts()
        return self

    def stop(self) -> SustainabilityReport:
        if self._start_monotonic is None:
            raise RuntimeError("SustainabilityTracker.stop() called before start()")
        duration = max(0.0, time.monotonic() - self._start_monotonic)
        end_gpu_watts = _read_gpu_power_watts()

        if self._start_gpu_watts is None and end_gpu_watts is None:
            if self.gpu_count > 0:
                gpu_watts = FALLBACK_GPU_TDP_WATTS * self.gpu_count
                method = "tdp-fallback"
            else:
                gpu_watts = 0.0
                method = "cpu-only"
        else:
            samples = [w for w in (self._start_gpu_watts, end_gpu_watts)
                       if w is not None]
            gpu_watts = sum(samples) / len(samples)
            method = "measured" if len(samples) == 2 else "partial-sample"

        total_watts = gpu_watts + _cpu_watts()
        energy_kwh = total_watts * duration / 3_600_000.0
        co2_kg = energy_kwh * self.carbon_intensity
        self._report = SustainabilityReport(
            duration_seconds=duration,
            energy_kwh=energy_kwh,
            co2_kg=co2_kg,
            hardware=self.hardware,
            carbon_intensity=self.carbon_intensity,
            gpu_count=self.gpu_count,
            method=method,
        )
        self._start_monotonic = None
        return self._report

    def __enter__(self) -> "SustainabilityTracker":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_monotonic is not None:
            self.stop()

    @property
    def report(self) -> Optional[SustainabilityReport]:
        return self._report
