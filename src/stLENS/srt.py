"""Spatial-graph-based SRT (Signal Robustness Test) perturbation.

stLENS's SRT perturbs matrix entries fully at random (entry-wise), which
can't distinguish spatially-structured signal from purely random
technical noise as sharply as a perturbation that respects spatial
structure. This module partitions spots into spatial blocks and permutes
values block-wise (destroying large-scale spatial coherence while keeping
local structure), and scores candidate PCs by Moran's I against the
shared spatial graph as an additional signal-vs-noise criterion.
"""

from __future__ import annotations

import numpy as np

from .coreset import knn_pseudobulk
from .eigensolver import wishart_top_k
from .spatial_graph import build_spatial_knn_graph


def morans_i(values, graph):
    """Global Moran's I of a single vector `values` (one entry per node)
    against a sparse weighted adjacency `graph`."""
    return float(morans_i_batch(np.asarray(values, dtype=np.float64)[:, None], graph)[0])


def morans_i_batch(vectors, graph):
    """Moran's I for many candidate vectors at once via one sparse matvec.

    `vectors`: (n_nodes, n_candidates). Returns an array of length
    n_candidates.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n = vectors.shape[0]
    z = vectors - vectors.mean(axis=0, keepdims=True)
    w_sum = graph.sum()

    if w_sum == 0:
        return np.zeros(vectors.shape[1])

    numerator = np.einsum("ij,ij->j", z, graph @ z)
    denominator = np.sum(z ** 2, axis=0)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return (n / w_sum) * (numerator / denominator)


def spatial_block_permute(X, coords=None, block_size=50, rng=None, groups=None):
    """Permute rows of `X` block-wise: spots within the same spatial block
    are shuffled among themselves, keeping local structure but destroying
    spatially-coherent signal at scales larger than a block."""
    rng = np.random.default_rng(rng)
    if groups is None:
        if coords is None:
            raise ValueError("either `coords` or precomputed `groups` must be provided")
        groups = knn_pseudobulk(coords, group_size=block_size)

    X_perm = np.array(X, copy=True)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        perm = rng.permutation(idx)
        X_perm[idx] = X[perm]
    return X_perm


def spatial_srt(backend, X, coords, k_signal, n_perturb=20, block_size=50, knn_k=15,
                 moran_threshold=0.1, survival_fraction=0.5, rng=None):
    """Simplified spatial-graph SRT.

    Computes the top-`k_signal` eigenvectors of the (spot x spot) Wishart
    matrix `X @ X.T` -- via whichever of `X @ X.T` / `X.T @ X` is smaller,
    so this stays memory-safe at large spot counts -- scores them by
    Moran's I against the spatial graph, then repeats with `n_perturb`
    spatial block permutations of `X`. A component only counts as
    confirmed spatial signal if its Moran's I clears `moran_threshold` on
    the real data AND still clears it on at least `survival_fraction` of
    the perturbed copies (pure noise loses its spatial structure under
    block permutation; real spatial signal does not).

    Returns `(n_signal_confirmed, base_moran_scores)`.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=np.float64)

    graph = build_spatial_knn_graph(coords, k=knn_k)
    groups = knn_pseudobulk(coords, group_size=block_size)

    _, evecs = wishart_top_k(backend, X, k_signal)
    evecs_host = backend.to_numpy(evecs)
    base_scores = morans_i_batch(evecs_host, graph)

    survival_counts = np.zeros(k_signal)
    for _ in range(n_perturb):
        X_perm = spatial_block_permute(X, groups=groups, rng=rng)
        _, evecs_p = wishart_top_k(backend, X_perm, k_signal)
        scores_p = morans_i_batch(backend.to_numpy(evecs_p), graph)
        survival_counts += scores_p > moran_threshold

    confirmed = int(np.sum(
        (base_scores > moran_threshold) & (survival_counts >= n_perturb * survival_fraction)
    ))
    return confirmed, base_scores
