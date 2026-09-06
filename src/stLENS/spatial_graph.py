"""Shared approximate spatial KNN graph.

Used by spatially-aware normalization, spatial-graph SRT perturbation, and
spatial coreset compression. Built once via a KD-tree (not full pairwise
distances) so it stays fast at million-spot scale.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree


def build_spatial_knn_graph(coords, k=15, symmetric=True):
    """Approximate spatial KNN graph via a KD-tree.

    Returns a scipy.sparse CSR adjacency matrix of shape (n, n) with edge
    weights = euclidean distance, self-loops excluded. `symmetric=True`
    makes the graph undirected (A = max(A, A.T)) since KNN is not
    naturally symmetric (i being in j's KNN doesn't imply the reverse).
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return csr_matrix((n, n))

    tree = cKDTree(coords)
    dists, idx = tree.query(coords, k=k + 1)  # includes self at distance 0
    dists, idx = dists[:, 1:], idx[:, 1:]  # drop self

    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    data = dists.ravel()
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))

    if symmetric:
        graph = graph.maximum(graph.T)

    return graph


def to_similarity_weights(graph, bandwidth=None):
    """Convert a distance-weighted graph to a similarity-weighted graph via
    a Gaussian kernel: closer spots get higher weight. `bandwidth` defaults
    to the median nonzero edge distance."""
    graph = graph.tocsr(copy=True)
    data = graph.data
    if data.size == 0:
        return graph

    if bandwidth is None:
        bandwidth = np.median(data)
        if bandwidth <= 0:
            bandwidth = 1.0

    graph.data = np.exp(-(data ** 2) / (2 * bandwidth ** 2))
    return graph


def row_normalize(graph):
    """Row-normalize a sparse weighted graph so each row sums to 1."""
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    return diags(inv) @ graph
