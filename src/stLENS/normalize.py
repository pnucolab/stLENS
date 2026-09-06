"""Spatially-aware normalization.

Corrects each spot's normalization scale using the local background of
spatially nearby spots (count density, tissue-region batch effects)
instead of treating every spot independently, as plain stLENS does.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from .spatial_graph import build_spatial_knn_graph, to_similarity_weights


def local_background(X, coords, k=15, graph=None):
    """Per-spot local background: the similarity-weighted mean total count
    of a spot's spatial neighbors, via the shared spatial KNN graph."""
    if graph is None:
        graph = to_similarity_weights(build_spatial_knn_graph(coords, k=k))

    totals = np.asarray(X.sum(axis=1)).ravel() if issparse(X) else np.asarray(X).sum(axis=1)

    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    weighted = graph @ totals
    background = np.zeros_like(weighted)
    nz = row_sums > 0
    background[nz] = weighted[nz] / row_sums[nz]
    background[~nz] = totals[~nz]  # isolated spots: fall back to their own total
    return background


def spatial_normalize(X, coords, k=15, eps=1e-8):
    """L1-scale -> log1p -> z-score normalization, where each spot's scale
    factor is its local spatial background rather than only its own total
    count -- this is what lets normalization correct for smooth,
    non-biological spatial gradients (batch effects, tissue thickness)
    instead of just per-spot depth."""
    totals = np.asarray(X.sum(axis=1)).ravel() if issparse(X) else np.asarray(X).sum(axis=1)
    background = local_background(X, coords, k=k)
    scale = np.where(background > eps, background, totals + eps)

    if issparse(X):
        from scipy.sparse import diags

        X_scaled = diags(1.0 / scale) @ X
        X_scaled = X_scaled.tocsr()
        X_scaled.data = np.log1p(X_scaled.data)
        X_dense = np.asarray(X_scaled.todense())
    else:
        X_dense = np.log1p(np.asarray(X, dtype=np.float64) / scale[:, None])

    mean = X_dense.mean(axis=0)
    std = X_dense.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (X_dense - mean) / std
