import numpy as np
import pytest

from stLENS.backend import Backend
from stLENS.eigensolver import top_k_eigh, wishart_top_k


def _random_wishart(n, d, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float64)
    return x @ x.T


def _align_sign(v_ref, v_test):
    signs = np.sign(np.sum(v_ref * v_test, axis=0))
    signs[signs == 0] = 1.0
    return v_test * signs


@pytest.mark.parametrize("n,k", [(50, 5), (200, 10), (200, 30)])
def test_top_k_matches_full_eigh(n, k):
    backend = Backend(use_gpu=False)
    a = backend.asarray(_random_wishart(n, n // 2, seed=n + k))
    full_evals, full_evecs = np.linalg.eigh(a)
    ref_evals, ref_evecs = full_evals[-k:], full_evecs[:, -k:]
    evals, evecs = top_k_eigh(backend, a, k)
    np.testing.assert_allclose(evals, ref_evals, rtol=1e-6, atol=1e-6)
    evecs_aligned = _align_sign(ref_evecs, np.asarray(evecs))
    np.testing.assert_allclose(evecs_aligned, ref_evecs, atol=1e-5)


def test_top_k_eigenvectors_are_orthonormal():
    backend = Backend(use_gpu=False)
    a = backend.asarray(_random_wishart(100, 50, seed=1))
    _, evecs = top_k_eigh(backend, a, k=8)
    gram = evecs.T @ evecs
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-6)


def test_k_close_to_n_falls_back_to_full_eigh():
    backend = Backend(use_gpu=False)
    a = backend.asarray(_random_wishart(20, 10, seed=2))
    full_evals, full_evecs = np.linalg.eigh(a)
    evals, evecs = top_k_eigh(backend, a, k=19)
    np.testing.assert_allclose(evals, full_evals[-19:], rtol=1e-8)
    evecs_aligned = _align_sign(full_evecs[:, -19:], np.asarray(evecs))
    np.testing.assert_allclose(evecs_aligned, full_evecs[:, -19:], atol=1e-6)


def test_k_zero_returns_empty():
    backend = Backend(use_gpu=False)
    a = backend.asarray(_random_wishart(30, 15, seed=3))
    evals, evecs = top_k_eigh(backend, a, k=0)
    assert evals.shape == (0,)
    assert evecs.shape == (30, 0)


def test_eigenvalues_sorted_ascending():
    backend = Backend(use_gpu=False)
    a = backend.asarray(_random_wishart(80, 40, seed=4))
    evals, _ = top_k_eigh(backend, a, k=12)
    assert np.all(np.diff(evals) >= -1e-9)


def _reference_wishart_top_k(X, k):
    """Brute-force reference: always form the spot x spot Gram matrix directly."""
    gram = X @ X.T
    evals, evecs = np.linalg.eigh(gram)
    return evals[-k:], evecs[:, -k:]


def test_wishart_top_k_matches_reference_when_more_spots_than_genes():
    backend = Backend(use_gpu=False)
    rng = np.random.default_rng(10)
    X = rng.standard_normal((150, 30))  # n_spots > n_genes -> takes the gene-space branch
    k = 6

    ref_evals, ref_evecs = _reference_wishart_top_k(X, k)
    evals, evecs = wishart_top_k(backend, X, k)

    np.testing.assert_allclose(np.asarray(evals), ref_evals, rtol=1e-5, atol=1e-6)
    evecs_aligned = _align_sign(ref_evecs, np.asarray(evecs))
    np.testing.assert_allclose(evecs_aligned, ref_evecs, atol=1e-4)


def test_wishart_top_k_matches_reference_when_more_genes_than_spots():
    backend = Backend(use_gpu=False)
    rng = np.random.default_rng(11)
    X = rng.standard_normal((30, 150))  # n_genes > n_spots -> takes the direct spot-space branch
    k = 6

    ref_evals, ref_evecs = _reference_wishart_top_k(X, k)
    evals, evecs = wishart_top_k(backend, X, k)

    np.testing.assert_allclose(np.asarray(evals), ref_evals, rtol=1e-5, atol=1e-6)
    evecs_aligned = _align_sign(ref_evecs, np.asarray(evecs))
    np.testing.assert_allclose(evecs_aligned, ref_evecs, atol=1e-4)


def test_wishart_top_k_stays_memory_safe_at_large_spot_count():
    """n_spots >> n_genes: must decompose the small gene x gene Gram
    matrix, never materialize the huge spot x spot one."""
    backend = Backend(use_gpu=False)
    rng = np.random.default_rng(12)
    n_spots, n_genes = 50_000, 100
    X = rng.standard_normal((n_spots, n_genes)).astype(np.float32)

    evals, evecs = wishart_top_k(backend, X, k=5)
    assert evecs.shape == (n_spots, 5)
    assert np.isfinite(np.asarray(evals)).all()
