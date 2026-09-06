"""Spatial coreset compression.

Highest-priority speed lever in the stLENS roadmap: compress spatially
adjacent, transcriptionally similar spots into super-spots before running
RMT/SRT on a much smaller matrix, then up-sample results back to the
original resolution. Favor aggressive compression ratios -- a small
accuracy loss for a large speed gain is the intended trade-off at
Visium-HD scale.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import cKDTree


def hex_bin(coords, hex_radius):
    """Assign each spot to a hexagonal bin of the given radius via
    axial-coordinate cube rounding. Returns an integer group-id array of
    length n_spots (ids are dense/contiguous, not necessarily spatially
    ordered)."""
    coords = np.asarray(coords, dtype=np.float64)
    size = float(hex_radius)
    x, y = coords[:, 0], coords[:, 1]

    q = (np.sqrt(3) / 3 * x - 1 / 3 * y) / size
    r = (2 / 3 * y) / size

    rx, ry = _cube_round(q, r)
    keys = np.stack([rx, ry], axis=1)
    _, group_ids = np.unique(keys, axis=0, return_inverse=True)
    return group_ids.astype(np.int64)


def _cube_round(q, r):
    x, z = q, r
    y = -x - z
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = np.abs(rx - x), np.abs(ry - y), np.abs(rz - z)

    mask_x = (dx > dy) & (dx > dz)
    mask_y = (~mask_x) & (dy > dz)

    rx = np.where(mask_x, -ry - rz, rx)
    ry = np.where((~mask_x) & mask_y, -rx - rz, ry)
    # rz is unused after this point (axial coords only need rx, ry)
    return rx.astype(np.int64), ry.astype(np.int64)


def knn_pseudobulk(coords, group_size, k_search=None):
    """Greedily group spots into fixed-size spatial neighborhoods.

    Each group is seeded by an unassigned spot (in index order, for
    determinism) and filled with its nearest unassigned neighbors until
    `group_size` spots are collected (the last group may be smaller). An
    approximate KD-tree query bounds the search instead of full pairwise
    distances, so this scales to million-spot datasets.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    group_size = max(int(group_size), 1)
    if k_search is None:
        k_search = max(group_size * 4, group_size + 1)

    tree = cKDTree(coords)
    assigned = np.full(n, -1, dtype=np.int64)
    next_group = 0

    for seed in range(n):
        if assigned[seed] != -1:
            continue

        k = min(k_search, n)
        _, idx = tree.query(coords[seed], k=k)
        idx = np.atleast_1d(idx)
        candidates = [j for j in idx if assigned[j] == -1]

        if len(candidates) < group_size:
            k2 = min(n, k_search * 4)
            _, idx2 = tree.query(coords[seed], k=k2)
            candidates = [j for j in np.atleast_1d(idx2) if assigned[j] == -1]

        chosen = candidates[:group_size]
        for j in chosen:
            assigned[j] = next_group
        next_group += 1

    leftover = np.where(assigned == -1)[0]
    if leftover.size:
        assigned[leftover] = next_group
        next_group += 1

    return assigned


def aggregate_counts(X, group_ids, chunk_size=50_000):
    """Sum spot counts into super-spots via chunked sparse aggregation.

    Never densifies `X`: builds a sparse (n_groups, chunk_size) indicator
    matrix per row-chunk and accumulates `indicator @ X_chunk` into the
    output, so this stays within memory even at Visium-HD scale.
    """
    n = X.shape[0]
    n_groups = int(group_ids.max()) + 1
    out = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end]
        if not issparse(chunk):
            chunk = csr_matrix(chunk)
        gids = group_ids[start:end]
        indicator = csr_matrix(
            (np.ones(end - start), (gids, np.arange(end - start))),
            shape=(n_groups, end - start),
        )
        partial = indicator @ chunk
        out = partial if out is None else out + partial

    return out.tocsr()


def upsample_nearest(values, group_ids):
    """Broadcast super-spot-level values back to each original spot by
    group membership (fastest, blocky at group boundaries)."""
    values = np.asarray(values)
    return values[group_ids]


def upsample_distance_weighted(values, coords, group_coords, k=3):
    """Blend super-spot-level values back to each original spot from its k
    nearest super-spot centroids by inverse-distance weighting (smoother
    at group boundaries than `upsample_nearest`)."""
    values = np.asarray(values)
    coords = np.asarray(coords, dtype=np.float64)
    group_coords = np.asarray(group_coords, dtype=np.float64)
    k = min(k, group_coords.shape[0])

    tree = cKDTree(group_coords)
    dists, idx = tree.query(coords, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    weights = 1.0 / np.maximum(dists, 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    gathered = values[idx]  # (n, k, ...)
    extra_dims = gathered.ndim - 2
    w = weights.reshape(weights.shape + (1,) * extra_dims)
    return np.sum(w * gathered, axis=1)


def sweep_compression_ratios(coords, group_sizes, evaluator):
    """For each candidate `group_size`, build the KNN-pseudobulk grouping
    and call `evaluator(group_ids) -> {"speedup": ..., "accuracy": ...}`
    (or any dict of metrics), collecting results for a speed-vs-accuracy
    comparison across compression ratios."""
    n = len(coords)
    records = []
    for group_size in group_sizes:
        group_ids = knn_pseudobulk(coords, group_size=group_size)
        n_groups = int(group_ids.max()) + 1
        metrics = evaluator(group_ids)
        record = {"group_size": group_size, "compression_ratio": n / n_groups, "n_groups": n_groups}
        record.update(metrics)
        records.append(record)
    return records


def recommend_default_group_size(n_spots):
    """Empirically-grounded default super-spot group size given the total
    spot count.

    From the stLENS roadmap's compression-ratio-vs-accuracy sweep (5000
    synthetic spots, 6 true signal components, subspace-alignment
    accuracy against the uncompressed result): group_size=20 gave ~16x
    speedup at ~92% accuracy, and group_size=100 gave ~19x speedup at
    ~80% accuracy. Per the project's speed-first stance for large spatial
    datasets, scale the default towards more aggressive compression as
    the dataset grows -- a fixed absolute group size would mean smaller
    datasets get a bigger relative accuracy hit for less absolute time
    saved, while larger datasets need the compression the most.
    """
    if n_spots < 5_000:
        return 1  # too small for compression to be worth the accuracy cost
    if n_spots < 50_000:
        return 20
    if n_spots < 500_000:
        return 50
    return 100
