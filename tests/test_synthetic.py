import numpy as np

from stLENS.synthetic import generate_spatial_dataset


def test_output_shapes_and_true_pc_count():
    X, coords, true_k, loadings = generate_spatial_dataset(
        n_spots=500, n_genes=80, n_signal_components=6, seed=0
    )
    assert X.shape == (500, 80)
    assert coords.shape == (500, 2)
    assert true_k == 6
    assert loadings.shape == (6, 80)
    assert (X >= 0).all()


def test_eigenvalue_gap_separates_signal_from_noise():
    X, _, true_k, _ = generate_spatial_dataset(
        n_spots=600, n_genes=100, n_signal_components=5, noise_level=0.3, seed=1
    )
    X_centered = X - X.mean(axis=0)
    cov = X_centered @ X_centered.T
    evals = np.linalg.eigvalsh(cov)
    evals_desc = evals[::-1]

    signal_evals = evals_desc[:true_k]
    noise_evals = evals_desc[true_k : true_k + 10]
    assert signal_evals.min() > noise_evals.max() * 2
