"""Synthetic spatial transcriptomics data with known signal/noise
eigenstructure and a known spatial autocorrelation length scale, so
stLENS's new approximations (truncated eigensolver, spatial SRT, coreset
compression) can be checked against ground truth rather than just real
data where the "true" answer is unknown.
"""

from __future__ import annotations

import numpy as np


def generate_spatial_dataset(
    n_spots=2000,
    n_genes=200,
    n_signal_components=5,
    spatial_length_scale=5.0,
    noise_level=1.0,
    grid=True,
    seed=0,
):
    """Generate a synthetic spot x gene matrix with `n_signal_components`
    true spatially-coherent signal components (smooth over
    `spatial_length_scale`) plus iid Gaussian technical noise, so the true
    number of "real" PCs is known exactly.

    Returns `(X, coords, true_pc_count, gene_loadings)`.
    """
    rng = np.random.default_rng(seed)

    if grid:
        side = int(np.ceil(np.sqrt(n_spots)))
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)[:n_spots]
    else:
        coords = rng.uniform(0, np.sqrt(n_spots), size=(n_spots, 2))

    # Smooth spatial factors: random low-frequency sinusoids evaluated at coords.
    factors = np.zeros((n_spots, n_signal_components))
    for c in range(n_signal_components):
        freq = rng.uniform(0.5, 2.0, size=2) / spatial_length_scale
        phase = rng.uniform(0, 2 * np.pi)
        factors[:, c] = np.sin(coords @ freq + phase)

    gene_loadings = rng.standard_normal((n_signal_components, n_genes))
    signal = factors @ gene_loadings

    noise = noise_level * rng.standard_normal((n_spots, n_genes))
    X = signal + noise
    X = np.clip(X - X.min(), 0, None)  # keep values non-negative, count-like

    return X, coords, n_signal_components, gene_loadings


def generate_spatial_dataset_with_spatial_noise(
    n_spots=2000,
    n_genes=200,
    n_signal_components=5,
    spatial_length_scale=8.0,
    signal_noise_level=0.3,
    technical_noise_level=1.0,
    technical_noise_knn_k=10,
    technical_noise_smoothing_steps=1,
    grid=True,
    seed=0,
):
    """Like `generate_spatial_dataset`, but the technical noise is
    spatially autocorrelated PER GENE (each gene independently smoothed
    over the spatial KNN graph, no cross-gene structure imposed) instead
    of i.i.d. across spots.

    This mimics realistic spatial transcriptomics artifacts (diffusion
    between neighboring spots, shared local capture efficiency, tissue
    batch effects) that correlate a single gene's noise across nearby
    spots without creating any real shared multi-gene signal. This is
    exactly the regime where classical i.i.d.-null RMT/parallel-analysis
    thresholds are expected to OVER-COUNT signal components (see
    `spatial_null.py`): each gene's noise, though independent of every
    other gene's noise, has a smaller effective sample size than
    n_spots, inflating the top eigenvalues of the pure-noise part of the
    Wishart matrix beyond the classical i.i.d. prediction.

    Returns `(X, coords, true_pc_count, gene_loadings)` -- same contract
    as `generate_spatial_dataset`, so both can be dropped into the same
    validation code.
    """
    from .spatial_graph import build_spatial_knn_graph, row_normalize, to_similarity_weights

    rng = np.random.default_rng(seed)

    if grid:
        side = int(np.ceil(np.sqrt(n_spots)))
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)[:n_spots]
    else:
        coords = rng.uniform(0, np.sqrt(n_spots), size=(n_spots, 2))

    factors = np.zeros((n_spots, n_signal_components))
    for c in range(n_signal_components):
        freq = rng.uniform(0.5, 2.0, size=2) / spatial_length_scale
        phase = rng.uniform(0, 2 * np.pi)
        factors[:, c] = np.sin(coords @ freq + phase)

    gene_loadings = rng.standard_normal((n_signal_components, n_genes))
    signal = factors @ gene_loadings
    signal_noise = signal_noise_level * rng.standard_normal((n_spots, n_genes))

    graph = build_spatial_knn_graph(coords, k=technical_noise_knn_k)
    S = row_normalize(to_similarity_weights(graph))
    for _ in range(technical_noise_smoothing_steps - 1):
        S = S @ S

    Z = rng.standard_normal((n_spots, n_genes))
    Z_smooth = S @ Z
    col_std = Z_smooth.std(axis=0)
    col_std = np.where(col_std < 1e-12, 1.0, col_std)
    technical_noise = (Z_smooth / col_std) * technical_noise_level

    X = signal + signal_noise + technical_noise
    X = np.clip(X - X.min(), 0, None)

    return X, coords, n_signal_components, gene_loadings
