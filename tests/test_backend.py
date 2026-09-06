import numpy as np
import pytest

from stLENS.backend import Backend, get_backend


def test_cpu_fallback_when_no_gpu_requested():
    backend = Backend(use_gpu=False)
    assert backend.is_gpu is False
    assert backend.xp is np


def test_auto_detect_does_not_raise():
    backend = get_backend(refresh=True)
    assert backend.xp is not None


def test_force_gpu_without_cuda_raises_clean_error(monkeypatch):
    monkeypatch.setattr("stLENS.backend._cupy_is_usable", lambda: False)
    with pytest.raises(RuntimeError):
        Backend(use_gpu=True)


def test_env_var_forces_cpu(monkeypatch):
    monkeypatch.setenv("STLENS_FORCE_CPU", "1")
    monkeypatch.setattr("stLENS.backend._cupy_is_usable", lambda: True)
    backend = Backend()
    assert backend.is_gpu is False


def test_asarray_and_to_numpy_roundtrip():
    backend = Backend(use_gpu=False)
    x = backend.asarray([1.0, 2.0, 3.0])
    host = backend.to_numpy(x)
    assert isinstance(host, np.ndarray)
    np.testing.assert_array_equal(host, [1.0, 2.0, 3.0])


def test_memory_pool_helpers_are_noops_on_cpu():
    backend = Backend(use_gpu=False)
    backend.free_all_blocks()
    pool = backend.get_default_memory_pool()
    pool.free_all_blocks()
    assert backend.device_count() == 0
    assert backend.free_bytes_and_total() == (0, 0)


def test_scoped_memory_context_manager_on_cpu():
    backend = Backend(use_gpu=False)
    with backend.scoped_memory():
        x = backend.asarray(np.eye(3))
        assert x.shape == (3, 3)


def test_out_of_memory_error_never_raised_on_cpu_but_is_catchable():
    backend = Backend(use_gpu=False)
    try:
        raise backend.OutOfMemoryError("simulated")
    except backend.OutOfMemoryError:
        caught = True
    assert caught


def test_get_backend_singleton_reuses_instance():
    b1 = get_backend(use_gpu=False)
    b2 = get_backend()
    assert b1 is b2


def test_real_gpu_backend_when_cupy_installed():
    """On a machine with cupy actually installed and a real CUDA device,
    the GPU path must work end to end (this test is skipped otherwise)."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device visible")

    backend = Backend(use_gpu=True)
    assert backend.is_gpu is True
    x = backend.asarray(np.eye(4))
    assert isinstance(x, cupy.ndarray)
    host = backend.to_numpy(x)
    np.testing.assert_array_equal(host, np.eye(4))
    backend.free_all_blocks()  # must not raise
