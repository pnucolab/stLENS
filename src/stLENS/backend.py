"""Array backend abstraction for stLENS.

stLENS hard-fails at import time if CuPy/a GPU is not present (`import cupy`
at module scope, raising ImportError immediately). That makes CPU-only
development, CI, and small-data usage impossible.

This module provides a single `Backend` object that:
- uses CuPy when it is importable AND a CUDA device is actually usable,
- otherwise transparently falls back to NumPy,
- exposes the handful of GPU-only operations stLENS relies on (memory pool
  management, device selection, the CuPy out-of-memory exception) as
  no-ops / CPU-safe equivalents, so calling code does not need to branch on
  `is_gpu` for those either.

Usage:
    from stLENS.backend import get_backend
    backend = get_backend()          # auto-detect
    xp = backend.xp                  # numpy or cupy module
    x = backend.asarray(some_array)
    ... xp.linalg.eigh(x) ...
    x_host = backend.to_numpy(x)     # always returns a numpy array

Force a specific backend (useful for tests / benchmarking):
    get_backend(use_gpu=False)
    get_backend(use_gpu=True)        # raises RuntimeError if no usable GPU

The STLENS_FORCE_CPU=1 environment variable forces the CPU path even if a
GPU is available, without changing call sites.
"""

from __future__ import annotations

import os
from contextlib import contextmanager


class _NullMemoryPool:
    """No-op stand-in for cupy's MemoryPool on the CPU backend."""

    def free_all_blocks(self):
        pass

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0


class _NullDevice:
    """No-op stand-in for cupy.cuda.Device on the CPU backend."""

    def __init__(self, device_id=0):
        self.id = device_id

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def use(self):
        return self


def _cupy_is_usable():
    """True only if cupy imports AND a CUDA device actually responds.

    A bare `import cupy` can succeed even when no driver/device is present;
    the actual failure often only surfaces on first device access. Probing
    device count here means callers get a clean CPU fallback instead of a
    crash deep inside the SRT/RMT loops.
    """
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        return True
    except Exception:
        return False


class Backend:
    """Uniform array-op interface over numpy (CPU) or cupy (GPU)."""

    def __init__(self, use_gpu: bool | None = None):
        force_cpu = os.environ.get("STLENS_FORCE_CPU", "").lower() in ("1", "true", "yes")

        if use_gpu is None:
            use_gpu = (not force_cpu) and _cupy_is_usable()
        elif use_gpu and not _cupy_is_usable():
            raise RuntimeError(
                "use_gpu=True was requested but CuPy / a CUDA device is not usable. "
                "Install cupy-cuda11x or cupy-cuda12x and ensure a GPU is visible, "
                "or call get_backend(use_gpu=False) / unset STLENS_FORCE_CPU."
            )

        self.is_gpu = bool(use_gpu)

        if self.is_gpu:
            import cupy as cp

            self.xp = cp
            self.OutOfMemoryError = cp.cuda.memory.OutOfMemoryError
            self.ndarray = cp.ndarray
        else:
            import numpy as np

            self.xp = np
            # numpy never raises a CuPy OOM error; alias to a class that is
            # never actually raised so `except backend.OutOfMemoryError:`
            # is always a valid, harmless no-op on CPU.
            self.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
            self.ndarray = np.ndarray

    # -- host/device transfer -------------------------------------------------

    def asarray(self, x, dtype=None):
        return self.xp.asarray(x, dtype=dtype) if dtype is not None else self.xp.asarray(x)

    def to_numpy(self, x):
        """Always returns a numpy array, regardless of active backend."""
        import numpy as np

        if self.is_gpu and isinstance(x, self.xp.ndarray):
            return self.xp.asnumpy(x)
        return np.asarray(x)

    # -- device / memory management (no-ops on CPU) ---------------------------

    def device_count(self):
        if self.is_gpu:
            return self.xp.cuda.runtime.getDeviceCount()
        return 0

    def device_context(self, device_id=0):
        """Context manager selecting a GPU device; a no-op on CPU."""
        if self.is_gpu:
            return self.xp.cuda.Device(device_id)
        return _NullDevice(device_id)

    def free_bytes_and_total(self, device_id=0):
        """(free_bytes, total_bytes) for the given device; (0, 0) on CPU."""
        if self.is_gpu:
            with self.xp.cuda.Device(device_id):
                return self.xp.cuda.runtime.memGetInfo()
        return (0, 0)

    def get_default_memory_pool(self):
        if self.is_gpu:
            return self.xp.get_default_memory_pool()
        return _NullMemoryPool()

    def get_default_pinned_memory_pool(self):
        if self.is_gpu:
            return self.xp.get_default_pinned_memory_pool()
        return _NullMemoryPool()

    def free_all_blocks(self):
        """Best-effort release of GPU memory pools; a no-op on CPU."""
        if self.is_gpu:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()

    @contextmanager
    def scoped_memory(self, device_id=0):
        """`with backend.scoped_memory():` selects the device (GPU) or does
        nothing (CPU), and always frees pooled memory blocks on exit."""
        with self.device_context(device_id):
            try:
                yield
            finally:
                self.free_all_blocks()


_default_backend: Backend | None = None


def get_backend(use_gpu: bool | None = None, refresh: bool = False) -> Backend:
    """Return the process-wide Backend, creating it on first call.

    `use_gpu=None` (default) auto-detects. Pass `refresh=True` to force
    re-detection (e.g. after changing STLENS_FORCE_CPU at runtime, or in
    tests that need a specific backend).
    """
    global _default_backend
    if _default_backend is None or refresh or use_gpu is not None:
        _default_backend = Backend(use_gpu=use_gpu)
    return _default_backend
