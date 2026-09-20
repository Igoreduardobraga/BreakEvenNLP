"""
RNGController Module.
Encapsulates multi-library RNG state capture, atomic context isolation,
stateful RNG streams, and experimental factor seed management.
"""

import os
import pickle
import random
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None


class RNGSnapshot:
    """
    Immutable capture of random states across Python, NumPy, PyTorch, and CUDA.
    """

    def __init__(
        self,
        python_state: Any,
        numpy_state: Any = None,
        torch_state: Any = None,
        cuda_state: Any = None,
    ):
        self.python_state = python_state
        self.numpy_state = numpy_state
        self.torch_state = torch_state
        self.cuda_state = cuda_state

    def restore(self) -> None:
        """Restores all captured RNG states safely."""
        random.setstate(self.python_state)

        if _HAS_NUMPY and self.numpy_state is not None:
            np.random.set_state(self.numpy_state)

        if _HAS_TORCH and self.torch_state is not None:
            torch.set_rng_state(self.torch_state)

        if _HAS_TORCH and self.cuda_state is not None:
            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available() is True:
                    torch.cuda.set_rng_state(self.cuda_state)
            except Exception:
                pass

    def __iter__(self):
        """Allows unpacking as (torch_state, cuda_state, numpy_state, python_state) for legacy compatibility."""
        yield self.torch_state
        yield self.cuda_state
        yield self.numpy_state
        yield self.python_state

    def __getitem__(self, index: int):
        return [self.torch_state, self.cuda_state, self.numpy_state, self.python_state][index]


class RNGStream:
    """
    A stateful, pausable stream of pseudo-random numbers.
    Maintains its own internal RNG state across activations without leaking
    to or from the surrounding global RNG.
    """

    def __init__(self, seed: int):
        self.seed = seed
        with RNGController.isolate(seed):
            self._state = RNGController.capture_state()
        self._outer_state: Optional[RNGSnapshot] = None

    def __enter__(self):
        self._outer_state = RNGController.capture_state()
        self._state.restore()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._state = RNGController.capture_state()
        if self._outer_state is not None:
            self._outer_state.restore()
            self._outer_state = None



class RNGController:
    """
    Deep module providing atomic seed isolation, multi-library RNG synchronization,
    and factor-based seed management.
    """

    @classmethod
    def capture_state(cls) -> RNGSnapshot:
        """Captures current RNG states across all available libraries."""
        py_state = random.getstate()
        np_state = np.random.get_state() if _HAS_NUMPY else None

        torch_state = None
        cuda_state = None
        if _HAS_TORCH:
            try:
                torch_state = torch.get_rng_state()
            except Exception:
                pass

            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available() is True:
                    cuda_state = torch.cuda.get_rng_state()
            except Exception:
                pass

        return RNGSnapshot(
            python_state=py_state,
            numpy_state=np_state,
            torch_state=torch_state,
            cuda_state=cuda_state,
        )

    @classmethod
    def seed_all(cls, seed: int) -> None:
        """Synchronously seeds Python random, NumPy, PyTorch CPU and CUDA."""
        random.seed(seed)

        if _HAS_NUMPY:
            np.random.seed(seed)

        if _HAS_TORCH:
            try:
                torch.manual_seed(seed)
                if hasattr(torch, "cuda") and torch.cuda.is_available() is True:
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

    @classmethod
    @contextmanager
    def isolate(cls, seed: Optional[int] = None):
        """
        Context manager that isolates random state.
        Restores previous RNG state on exit with guaranteed try/finally.
        """
        saved = cls.capture_state()
        try:
            if seed is not None:
                cls.seed_all(seed)
            yield cls
        finally:
            saved.restore()

    @classmethod
    def create_stream(cls, seed: int) -> RNGStream:
        """Creates a stateful, isolated RNG stream initialized with seed."""
        return RNGStream(seed=seed)

    FACTORS = (
        'label_choice',
        'sample_choice',
        'sample_order',
        'model_initialisation',
        'model_randomness',
    )

    @classmethod
    def build_factor_seeds(
        cls,
        base_seed: int,
        isolated_factor: Optional[str] = None,
        factor_seed: Optional[int] = None,
        fixed_seed: int = 42,
    ) -> Dict[str, int]:
        """
        Builds the dictionary of seeds for each experimental factor.
        If isolated_factor is None or 'golden_model'/'data_split', all factors share base_seed.
        If isolated_factor is specified, only that factor receives base_seed (or factor_seed),
        while all other factors receive fixed_seed.
        """
        if isolated_factor is None or isolated_factor in ('golden_model', 'data_split'):
            return {f: base_seed for f in cls.FACTORS}

        seeds = {}
        for f in cls.FACTORS:
            if f == isolated_factor:
                seeds[f] = factor_seed if factor_seed is not None else base_seed
            else:
                seeds[f] = fixed_seed
        return seeds

    @classmethod
    def generate_run_seeds(
        cls,
        rskf_seed: int,
        total_runs: int,
        cache_path: Optional[str] = None,
        regenerate: bool = False,
    ) -> List[int]:
        """
        Loads or deterministically generates total_runs seeds from rskf_seed.
        Optionally caches to disk at cache_path.
        """
        if cache_path and os.path.exists(cache_path) and not regenerate:
            try:
                with open(cache_path, 'rb') as f:
                    seeds = pickle.load(f)
                if len(seeds) == total_runs:
                    return seeds
            except Exception:
                pass

        with cls.isolate(rskf_seed):
            seeds = [random.randint(1, 100000) for _ in range(total_runs)]

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(seeds, f)
            except Exception:
                pass

        return seeds


