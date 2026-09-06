"""Auto spatial-scale calibration via subsampled variogram estimation.

Full pairwise semivariance is O(n^2) and infeasible at Visium-HD scale, so
this estimates the empirical semivariogram from a fixed-size random
subsample of spot pairs, then derives default radius/smoothing
hyperparameters for the normalization, SRT-graph, and coreset-binning
modules from the estimated spatial autocorrelation range -- replacing
manual tuning with a single data-driven scale parameter.
"""

from __future__ import annotations

import numpy as np


def subsampled_variogram(coords, values, n_pairs=20_000, n_bins=20, rng=None, max_distance=None):
    """Empirical semivariogram gamma(h) estimated from `n_pairs` random
    spot pairs (not all pairs), binned by distance `h`.

    Returns `(bin_centers, gamma)`; `gamma` is NaN in empty bins.
    """
    rng = np.random.default_rng(rng)
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = coords.shape[0]

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    dist = np.linalg.norm(coords[i] - coords[j], axis=1)
    sq_diff = 0.5 * (values[i] - values[j]) ** 2

    if max_distance is None:
        max_distance = np.quantile(dist, 0.9)

    bins = np.linspace(0, max_distance, n_bins + 1)
    bin_idx = np.digitize(dist, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    gamma = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (bin_idx == b) & (dist <= max_distance)
        if mask.sum() > 0:
            gamma[b] = sq_diff[mask].mean()

    return bin_centers, gamma


def estimate_autocorrelation_range(bin_centers, gamma, sill_fraction=0.9):
    """Distance at which the semivariogram first reaches `sill_fraction`
    of its sill (plateau) -- a fast heuristic for the practical spatial
    autocorrelation range, no nonlinear model fit required."""
    bin_centers = np.asarray(bin_centers)
    gamma = np.asarray(gamma)
    valid = ~np.isnan(gamma)

    if valid.sum() < 2:
        return float(bin_centers[-1]) if len(bin_centers) else 1.0

    centers, g = bin_centers[valid], gamma[valid]
    sill = np.max(g)
    if sill <= 0:
        return float(centers[-1])

    target = sill_fraction * sill
    above = np.where(g >= target)[0]
    if above.size == 0:
        return float(centers[-1])
    return float(centers[above[0]])


def calibrate_hyperparameters(autocorrelation_range):
    """Map an estimated spatial autocorrelation range to default
    hyperparameters for normalization / SRT-graph / coreset-binning."""
    r = max(float(autocorrelation_range), 1e-6)
    return {
        "normalization_knn_radius": r,
        "srt_block_radius": r,
        # compress at sub-autocorrelation scale so super-spots don't blend
        # distinct signal regions together.
        "coreset_hex_radius": r / 2.0,
    }
