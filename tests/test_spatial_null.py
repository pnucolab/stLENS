import numpy as np

from stLENS.backend import Backend
from stLENS.spatial_graph import build_spatial_knn_graph
from stLENS.spatial_null import (
    classical_iid_threshold,
    generate_spatial_null_matrix,
    spatial_mp_threshold,
    spatial_smoothing_operator,
)
from stLENS.synthetic import generate_spatial_dataset, generate_spatial_dataset_with_spatial_noise


def _grid_coords(side):
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def test_spatial_null_matrix_matches_target_variance():
    coords = _grid_coords(20)
    graph = build_spatial_knn_graph(coords, k=10)
    S = spatial_smoothing_operator(graph)
    target_var = np.array([1.0, 4.0, 9.0, 0.25] * 10)

    surrogate = generate_spatial_null_matrix(coords.shape[0], target_var, S, rng=0)
    np.testing.assert_allclose(surrogate.var(axis=0), target_var, rtol=0.35)


def test_spatial_null_matrix_has_spatial_autocorrelation():
    from stLENS.srt import morans_i

    coords = _grid_coords(20)
    graph = build_spatial_knn_graph(coords, k=10)
    S = spatial_smoothing_operator(graph, n_steps=2)

    surrogate = generate_spatial_null_matrix(coords.shape[0], np.ones(5), S, rng=1)
    scores = [morans_i(surrogate[:, c], graph) for c in range(5)]
    assert np.mean(scores) > 0.2  # clearly spatially structured, unlike i.i.d. noise


def test_pure_spatial_noise_no_signal_spatial_null_correctly_reports_near_zero():
    """Critical calibration test: data with NO real multi-gene signal at
    all, but strong per-gene spatial autocorrelation in the noise. A
    well-calibrated spatial null should report ~0 signal components."""
    backend = Backend(use_gpu=False)
    coords = _grid_coords(28)
    graph = build_spatial_knn_graph(coords, k=10)
    S = spatial_smoothing_operator(graph, n_steps=1)

    rng = np.random.default_rng(42)
    X = generate_spatial_null_matrix(coords.shape[0], np.ones(120), S, rng=rng)

    n_signal, _, _ = spatial_mp_threshold(
        backend, X, coords, alpha=0.05, n_surrogates=150, n_top=10, knn_k=10, smoothing_steps=1, rng=1
    )
    assert n_signal <= 1


def test_classical_null_overcounts_massively_on_pure_spatial_noise():
    """The key falsifiable claim: on data with spatially-correlated
    per-gene noise but NO real shared signal, the classical i.i.d. null
    (what plain RMT/parallel analysis assumes) badly over-counts "signal"
    components, while the spatial null correctly rejects nearly all of
    them. Empirically (see analysis/spatial-null-diagnostic), classical
    maxes out at n_top=10 "signal" components on data with zero true
    signal, while the spatial null reports 0-1.
    """
    backend = Backend(use_gpu=False)
    coords = _grid_coords(28)
    graph = build_spatial_knn_graph(coords, k=10)
    S = spatial_smoothing_operator(graph, n_steps=1)

    rng = np.random.default_rng(42)
    X = generate_spatial_null_matrix(coords.shape[0], np.ones(120), S, rng=rng)

    n_classical, _, _ = classical_iid_threshold(backend, X, alpha=0.05, n_surrogates=150, n_top=10, rng=1)
    n_spatial, _, _ = spatial_mp_threshold(
        backend, X, coords, alpha=0.05, n_surrogates=150, n_top=10, knn_k=10, smoothing_steps=1, rng=1
    )

    assert n_classical >= 8  # badly over-counts, near the tested ceiling
    assert n_spatial <= 1
    assert n_classical > n_spatial


def test_both_methods_recover_true_signal_when_noise_is_iid():
    """Sanity check: when technical noise really is i.i.d. (no spatial
    autocorrelation issue at all), the spatial null should not be overly
    conservative -- it should still recover the true signal count."""
    backend = Backend(use_gpu=False)
    X, coords, true_k, _ = generate_spatial_dataset(
        n_spots=500, n_genes=80, n_signal_components=3, spatial_length_scale=8.0,
        noise_level=0.4, seed=6,
    )
    n_spatial, _, _ = spatial_mp_threshold(backend, X, coords, alpha=0.05, n_surrogates=150, n_top=8, rng=7)
    assert abs(n_spatial - true_k) <= 1


def test_spatial_mp_threshold_beats_classical_on_realistic_mixed_dataset():
    """The headline validation: real shared signal PLUS realistic
    spatially-correlated per-gene technical noise (moderate strength,
    where the effect is reliable across seeds -- see
    analysis/spatial-null-diagnostic for the parameter sweep). The
    spatial-null threshold should land much closer to the true signal
    count than the classical i.i.d.-null threshold, averaged over
    several random seeds (a single seed can occasionally go either way).
    """
    backend = Backend(use_gpu=False)
    classical_errs, spatial_errs = [], []
    for seed in range(4):
        X, coords, true_k, _ = generate_spatial_dataset_with_spatial_noise(
            n_spots=784, n_genes=120, n_signal_components=4, spatial_length_scale=8.0,
            signal_noise_level=0.3, technical_noise_level=2.0, technical_noise_knn_k=10,
            technical_noise_smoothing_steps=2, seed=seed,
        )
        n_classical, _, _ = classical_iid_threshold(backend, X, alpha=0.05, n_surrogates=100, n_top=10, rng=seed)
        n_spatial, _, _ = spatial_mp_threshold(
            backend, X, coords, alpha=0.05, n_surrogates=100, n_top=10, knn_k=10, smoothing_steps=2, rng=seed
        )
        classical_errs.append(abs(n_classical - true_k))
        spatial_errs.append(abs(n_spatial - true_k))

    assert np.mean(spatial_errs) < np.mean(classical_errs)
    assert np.mean(spatial_errs) <= 1.0
