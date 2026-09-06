import numpy as np

from stLENS.backend import Backend
from stLENS.pipeline import run_spatial_pipeline
from stLENS.synthetic import generate_spatial_dataset, generate_spatial_dataset_with_spatial_noise


def test_run_spatial_pipeline_end_to_end_on_synthetic_data():
    X, coords, true_k, _ = generate_spatial_dataset(
        n_spots=400, n_genes=60, n_signal_components=4, spatial_length_scale=6.0,
        noise_level=0.5, seed=5,
    )
    backend = Backend(use_gpu=False)
    result = run_spatial_pipeline(X, coords, k_candidate=15, n_perturb=6, n_surrogates=60, backend=backend, rng=0)

    assert set(result.keys()) == {
        "X_pca", "X_pca_spatial", "n_signal_pcs", "moran_scores", "spatial_null_thresholds",
    }
    assert result["X_pca"].shape[0] == 400
    assert result["X_pca_spatial"].shape[0] == 400
    assert result["n_signal_pcs"] >= 1
    assert np.isfinite(result["X_pca"]).all()
    assert np.isfinite(result["X_pca_spatial"]).all()


def test_run_spatial_pipeline_does_not_massively_overcount_under_spatial_noise():
    """End-to-end check that the spatially-corrected threshold (not just
    the underlying spatial_mp_threshold function in isolation) keeps
    n_signal_pcs close to the true count even with realistic spatially-
    correlated technical noise."""
    X, coords, true_k, _ = generate_spatial_dataset_with_spatial_noise(
        n_spots=500, n_genes=80, n_signal_components=3, spatial_length_scale=8.0,
        signal_noise_level=0.3, technical_noise_level=1.5, technical_noise_knn_k=10,
        technical_noise_smoothing_steps=2, seed=1,
    )
    backend = Backend(use_gpu=False)
    result = run_spatial_pipeline(X, coords, k_candidate=12, n_perturb=6, n_surrogates=60, backend=backend, rng=0)
    assert result["n_signal_pcs"] <= true_k + 3
