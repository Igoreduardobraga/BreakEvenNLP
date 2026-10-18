"""
SustainabilityTracker: energy/carbon accounting per experiment fold.

Methodology (best available source first; the one used is reported in
`SustainabilityReport.method`, e.g. "gpu:nvml-energy;cpu:rapl"):

GPU, only the devices visible to this process (CUDA_VISIBLE_DEVICES, matched
by UUID when torch exposes it):
  1. nvml-energy      NVML cumulative energy counter, read at start and stop.
  2. nvml-sampled     NVML power draw sampled every SAMPLE_SECONDS, integrated.
  3. nvidia-smi-sampled  same, through `nvidia-smi` when pynvml is missing.
  4. tdp-fallback     FALLBACK_GPU_TDP_WATTS per visible device.

CPU (package-wide, like RAPL itself):
  1. rapl             /sys/class/powercap package energy counters.
  2. utilization-tdp  CPU_TDP_WATTS x CPU utilization, sampled and integrated.
                      CPU_TDP_WATTS defaults to 85 W (CodeCarbon's default for
                      unknown CPUs); set it to the real TDP for your machine.

energy_kwh = gpu_energy + cpu_energy. co2_kg = energy_kwh * carbon_intensity
(default 0.475 kg/kWh world average; override via arg or
CARBON_INTENSITY_KG_PER_KWH, e.g. Brazil ~0.074).

Usage:
    tracker = SustainabilityTracker()
    tracker.start()
    ... run fold ...
    report = tracker.stop()

Also usable as context manager: `with SustainabilityTracker() as t: ...`.
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

DEFAULT_CARBON_INTENSITY = 0.475
FALLBACK_GPU_TDP_WATTS = 250.0
DEFAULT_CPU_TDP_WATTS = 85.0
DEFAULT_SAMPLE_SECONDS = 1.0


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


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


def _strip_uuid(uuid: str) -> str:
    uuid = str(uuid).strip()
    return uuid[4:] if uuid.startswith("GPU-") else uuid


def _visible_gpus(count: int) -> List[str]:
    """Selectors for the GPUs this process sees: bare UUIDs or NVML indices.

    torch's UUIDs are exact; otherwise CUDA_VISIBLE_DEVICES is used as given
    (integer entries are taken as NVML indices, which matches CUDA's order
    under CUDA_DEVICE_ORDER=PCI_BUS_ID or on single-GPU machines).
    """
    if count == 0:
        return []
    try:
        import torch
        uuids = [getattr(torch.cuda.get_device_properties(i), "uuid", None) for i in range(count)]
        if all(u is not None for u in uuids):
            return [_strip_uuid(u) for u in uuids]
    except Exception:
        pass
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [_strip_uuid(t) for t in env.split(",") if t.strip()][:count]
    return [str(i) for i in range(count)]


class _NvmlGpus:
    """NVML handles for the visible GPUs; raises if NVML is unusable."""

    def __init__(self, selectors: List[str]):
        import pynvml
        self._nvml = pynvml
        pynvml.nvmlInit()
        self.handles = []
        for sel in selectors:
            if sel.isdigit():
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(int(sel)))
            else:
                self.handles.append(pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{sel}"))

    def energy_joules(self) -> Optional[float]:
        try:
            return sum(self._nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles) / 1000.0
        except Exception:
            return None

    def power_watts(self) -> Optional[float]:
        try:
            return sum(self._nvml.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000.0
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


def _nvidia_smi_power_fn(selectors: List[str]) -> Optional[Callable[[], Optional[float]]]:
    if shutil.which("nvidia-smi") is None:
        return None
    wanted = set(selectors)

    def read() -> Optional[float]:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,power.draw",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5, check=False,
            )
        except Exception:
            return None
        if res.returncode != 0:
            return None
        total, found = 0.0, False
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or (parts[0] not in wanted and _strip_uuid(parts[1]) not in wanted):
                continue
            try:
                total += float(parts[2])
                found = True
            except ValueError:
                pass
        return total if found else None

    return read


class _RaplCpu:
    """Package-level RAPL energy counters; raises if none are readable."""

    def __init__(self, root: str = "/sys/class/powercap"):
        self.domains = []
        for d in sorted(glob.glob(os.path.join(root, "intel-rapl:[0-9]*"))):
            if ":" in os.path.basename(d).split("intel-rapl:", 1)[1]:
                continue  # sub-domain (core/uncore/dram), already inside the package
            try:
                with open(os.path.join(d, "max_energy_range_uj")) as f:
                    max_range = int(f.read())
                self._read(d)
                self.domains.append((d, max_range))
            except (OSError, ValueError):
                continue
        if not self.domains:
            raise OSError("no readable RAPL package domain")

    @staticmethod
    def _read(domain: str) -> int:
        with open(os.path.join(domain, "energy_uj")) as f:
            return int(f.read())

    def snapshot(self) -> List[int]:
        return [self._read(d) for d, _ in self.domains]

    def joules_since(self, start: List[int]) -> float:
        total = 0
        for (d, max_range), s in zip(self.domains, start):
            delta = self._read(d) - s
            total += delta if delta >= 0 else delta + max_range  # counter wrapped
        return total / 1e6


def _cpu_utilization_fn() -> Optional[Callable[[], Optional[float]]]:
    try:
        import psutil
    except ImportError:
        return None
    psutil.cpu_percent(interval=None)  # prime: the first call has no reference point

    def read() -> Optional[float]:
        try:
            return psutil.cpu_percent(interval=None) / 100.0
        except Exception:
            return None

    return read


class _Integrator:
    """Trapezoidal integration of a watts reading sampled over time."""

    def __init__(self, read: Callable[[], Optional[float]]):
        self.read = read
        self.joules = 0.0
        self.samples = 0
        self._last = None

    def sample(self, now: float) -> None:
        watts = self.read()
        if watts is None:
            return
        if self._last is not None:
            t0, w0 = self._last
            self.joules += 0.5 * (w0 + watts) * (now - t0)
        self._last = (now, watts)
        self.samples += 1


@dataclass
class SustainabilityReport:
    duration_seconds: float
    energy_kwh: float
    co2_kg: float
    hardware: str
    carbon_intensity: float
    gpu_count: int
    method: str
    gpu_energy_kwh: float = 0.0
    cpu_energy_kwh: float = 0.0


class SustainabilityTracker:
    """Measures wall time and GPU/CPU energy for one fold."""

    def __init__(self, carbon_intensity: Optional[float] = None,
                 sample_seconds: Optional[float] = None):
        self.carbon_intensity = _resolve_carbon_intensity(carbon_intensity)
        self.sample_seconds = sample_seconds or _env_float("SUSTAINABILITY_SAMPLE_SECONDS", DEFAULT_SAMPLE_SECONDS)
        self.cpu_tdp_watts = _env_float("CPU_TDP_WATTS", DEFAULT_CPU_TDP_WATTS)
        self.hardware = _hardware_name()
        self.gpu_count = _cuda_device_count()
        self._start_monotonic: Optional[float] = None
        self._report: Optional[SustainabilityReport] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- source selection -------------------------------------------------
    def _setup_sources(self) -> None:
        selectors = _visible_gpus(self.gpu_count)
        self._nvml = None
        self._gpu_energy_start = None
        self._gpu = None
        self.gpu_method = "none"
        if selectors:
            try:
                self._nvml = _NvmlGpus(selectors)
            except Exception:
                self._nvml = None
            if self._nvml is not None:
                self._gpu_energy_start = self._nvml.energy_joules()
                if self._gpu_energy_start is not None:
                    self.gpu_method = "nvml-energy"
                else:
                    self._gpu = _Integrator(self._nvml.power_watts)
                    self.gpu_method = "nvml-sampled"
            else:
                smi = _nvidia_smi_power_fn(selectors)
                if smi is not None:
                    self._gpu = _Integrator(smi)
                    self.gpu_method = "nvidia-smi-sampled"
                else:
                    self.gpu_method = "tdp-fallback"

        self._rapl = None
        self._cpu = None
        try:
            self._rapl = _RaplCpu()
            self._rapl_start = self._rapl.snapshot()
            self.cpu_method = "rapl"
        except OSError:
            util = _cpu_utilization_fn()
            tdp = self.cpu_tdp_watts
            if util is not None:
                def cpu_watts() -> Optional[float]:
                    u = util()
                    return None if u is None else tdp * u
                self._cpu = _Integrator(cpu_watts)
                self.cpu_method = "utilization-tdp"
            else:
                self._cpu = _Integrator(lambda: tdp * 0.5)
                self.cpu_method = "tdp-half"

    def _sample_all(self) -> None:
        now = time.monotonic()
        for integ in (self._gpu, self._cpu):
            if integ is not None:
                integ.sample(now)

    def _run_sampler(self) -> None:
        while not self._stop_event.wait(self.sample_seconds):
            self._sample_all()

    # -- public API -------------------------------------------------------
    def start(self) -> "SustainabilityTracker":
        self._report = None
        self._setup_sources()
        self._start_monotonic = time.monotonic()
        self._sample_all()
        if self._gpu is not None or self._cpu is not None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_sampler, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> SustainabilityReport:
        if self._start_monotonic is None:
            raise RuntimeError("SustainabilityTracker.stop() called before start()")
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        self._sample_all()
        duration = max(0.0, time.monotonic() - self._start_monotonic)

        if self.gpu_method == "nvml-energy":
            end = self._nvml.energy_joules()
            gpu_joules = max(0.0, end - self._gpu_energy_start) if end is not None else 0.0
        elif self._gpu is not None and self._gpu.samples >= 2:
            gpu_joules = self._gpu.joules
        elif self.gpu_method != "none":
            self.gpu_method = "tdp-fallback"
            gpu_joules = FALLBACK_GPU_TDP_WATTS * self.gpu_count * duration
        else:
            gpu_joules = 0.0
        if self._nvml is not None:
            self._nvml.close()

        if self._rapl is not None:
            cpu_joules = self._rapl.joules_since(self._rapl_start)
        else:
            cpu_joules = self._cpu.joules

        gpu_kwh = gpu_joules / 3_600_000.0
        cpu_kwh = cpu_joules / 3_600_000.0
        energy_kwh = gpu_kwh + cpu_kwh
        self._report = SustainabilityReport(
            duration_seconds=duration,
            energy_kwh=energy_kwh,
            co2_kg=energy_kwh * self.carbon_intensity,
            hardware=self.hardware,
            carbon_intensity=self.carbon_intensity,
            gpu_count=self.gpu_count,
            method=f"gpu:{self.gpu_method};cpu:{self.cpu_method}",
            gpu_energy_kwh=gpu_kwh,
            cpu_energy_kwh=cpu_kwh,
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
