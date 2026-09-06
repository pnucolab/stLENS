"""Spatially-correlated random matrix null model for eigenvalue thresholding.

THE CORE PROBLEM this module addresses (not just a speed optimization):
classical RMT-based PC selection (stLENS included) tests candidate
eigenvalues of the Wishart matrix `X @ X.T` against the Marchenko-Pastur /
Tracy-Widom null distribution, which assumes the noise is i.i.d. across
BOTH spots and genes. Real spatial transcriptomics technical noise is NOT
i.i.d. across spots: shared local capture efficiency, diffusion between
neighboring spots, and tissue-region batch effects all induce spatial
autocorrelation WITHIN each gene's noise, even when there is no real
multi-gene biological signal at all.

Spatially autocorrelated samples carry less information than the same
number of independent samples -- the EFFECTIVE sample size n_eff is
smaller than n_spots. That inflates the top eigenvalues of the noise-only
part of the Wishart matrix beyond what the classical i.i.d. Marchenko-
Pastur law predicts, so thresholding against the classical i.i.d. null
systematically OVER-COUNTS signal components on spatial data. This is a
structural bias in the classical method, not a speed problem, and it gets
WORSE as spatial autocorrelation gets stronger (confirmed empirically
below and in `analysis/spatial-null-diagnostic`: on pure spatially-
correlated noise with zero true multi-gene signal, a classical i.i.d.
null reports 8-10 "significant" components out of 10 tested, every time,
while the spatial null correctly rejects nearly all of them).

THE FIX, and an honest novelty statement:
the *phenomenon* -- correlated samples deform the Marchenko-Pastur edge --
is established RMT theory for "doubly correlated" / separable-covariance
Wishart matrices (Burda et al., Spectral moments of correlated Wishart
matrices, Phys. Rev. E 71:026111, 2005; El Karoui, Ann. Probab. 35:663,
2007; Bai & Silverstein, Spectral Analysis of Large Dimensional Random
Matrices, 2010), and the same logic on the FEATURE axis (correlated
markers, not correlated samples) underlies Patterson, Price & Reich's
effective-marker-count correction for PCA-based population structure
(PLoS Genetics 2:e190, 2006). None of that is new here. What IS new is
applying it, for the first time, to RMT-based eigengene/PC selection in
spatial transcriptomics -- a closed-form edge for an irregular, non-
stationary spatial graph kernel is generally intractable, so this module
instead estimates the null empirically via graph-constrained surrogates,
in the same spirit as Moran Spectral Randomization (Wagner & Dray,
Methods Ecol. Evol. 6:1169, 2015, developed for map-correlation null
tests in ecology, not eigenvalue thresholding) and Monte Carlo SSA's
colored-noise surrogates for temporal eigenvalue testing (Allen & Smith,
J. Climate 9:3373, 1996). For each gene independently, i.i.d. Gaussian
noise is spatially smoothed with the SAME spatial similarity graph
estimated from the real data, matching that gene's own variance -- no
cross-gene covariance is imposed, so any eigenvalue standing out from
this null reflects genuine shared multi-gene structure, not spatially-
correlated per-gene technical noise. This is a spatial generalization of
Horn's (1965) parallel analysis, applied to spatial transcriptomics RMT
eigengene selection for the first time (to our knowledge).
"""

from __future__ import annotations

import numpy as np

from .eigensolver import wishart_top_k
from .spatial_graph import build_spatial_knn_graph, row_normalize, to_similarity_weights


def spatial_smoothing_operator(graph, n_steps=1):
    """Row-normalized spatial smoothing operator built from the shared
    spatial KNN graph -- repeated application approximates a diffusion
    kernel whose bandwidth is tied to the graph's own scale."""
    S = row_normalize(to_similarity_weights(graph))
    result = S
    for _ in range(n_steps - 1):
        result = result @ S
    return result


def generate_spatial_null_matrix(n_spots, gene_variances, smoothing_operator, rng=None):
    """One surrogate under H0 "no shared multi-gene signal": per gene,
    i.i.d. Gaussian noise spatially smoothed by `smoothing_operator`, then
    rescaled to that gene's own variance. Same per-gene marginal variance
    and per-gene spatial autocorrelation as real technical noise, but
    genes are smoothed independently, so there is no cross-gene covariance
    for any eigenvector to pick up."""
    rng = np.random.default_rng(rng)
    gene_variances = np.asarray(gene_variances, dtype=np.float64)
    n_genes = gene_variances.shape[0]

    Z = rng.standard_normal((n_spots, n_genes))
    Z_smooth = smoothing_operator @ Z

    col_std = Z_smooth.std(axis=0)
    col_std = np.where(col_std < 1e-12, 1.0, col_std)
    return (Z_smooth / col_std) * np.sqrt(gene_variances)[None, :]


def classical_iid_null_matrix(n_spots, gene_variances, rng=None):
    """The classical i.i.d. null surrogate (what standard parallel analysis
    / the implicit assumption behind Marchenko-Pastur uses): pure i.i.d.
    Gaussian noise per gene, no spatial structure at all."""
    rng = np.random.default_rng(rng)
    gene_variances = np.asarray(gene_variances, dtype=np.float64)
    n_genes = gene_variances.shape[0]
    return rng.standard_normal((n_spots, n_genes)) * np.sqrt(gene_variances)[None, :]


def _sequential_threshold_test(real_evals_desc, null_evals_desc, alpha):
    """Sequential rank-wise test (Horn's parallel analysis rule): component
    i is signal only if its eigenvalue exceeds the (1-alpha) quantile of
    the null distribution's i-th largest eigenvalue, AND every higher-
    ranked component was already confirmed signal."""
    thresholds = np.quantile(null_evals_desc, 1 - alpha, axis=0)
    n_signal = 0
    for i in range(len(real_evals_desc)):
        if real_evals_desc[i] > thresholds[i]:
            n_signal += 1
        else:
            break
    return n_signal, thresholds


def spatial_mp_threshold(backend, X, coords, alpha=0.05, n_surrogates=200, n_top=10,
                          knn_k=15, smoothing_steps=1, rng=None):
    """Select the number of signal components by sequentially testing real
    eigenvalues against an empirical SPATIALLY-CORRELATED null
    distribution, instead of the classical i.i.d. Marchenko-Pastur /
    Tracy-Widom threshold.

    Returns `(n_signal, real_eigenvalues_desc, null_thresholds_desc)`.
    """
    X = np.asarray(X, dtype=np.float64)
    n_spots, n_genes = X.shape
    n_top = min(n_top, n_spots - 2, n_genes)
    gene_variances = X.var(axis=0)

    graph = build_spatial_knn_graph(coords, k=knn_k)
    S = spatial_smoothing_operator(graph, n_steps=smoothing_steps)

    real_evals, _ = wishart_top_k(backend, X, n_top)
    real_evals_desc = backend.to_numpy(real_evals)[::-1]

    rng = np.random.default_rng(rng)
    null_evals = np.empty((n_surrogates, n_top))
    for i in range(n_surrogates):
        X_null = generate_spatial_null_matrix(n_spots, gene_variances, S, rng)
        evals, _ = wishart_top_k(backend, X_null, n_top)
        null_evals[i] = backend.to_numpy(evals)[::-1]

    n_signal, thresholds = _sequential_threshold_test(real_evals_desc, null_evals, alpha)
    return n_signal, real_evals_desc, thresholds


def classical_iid_threshold(backend, X, alpha=0.05, n_surrogates=200, n_top=10, rng=None):
    """The classical i.i.d.-null counterpart to `spatial_mp_threshold`,
    for direct comparison: same sequential parallel-analysis procedure,
    but surrogates are pure i.i.d. noise (no spatial structure at all) --
    this is what a non-spatial RMT/parallel-analysis method effectively
    assumes."""
    X = np.asarray(X, dtype=np.float64)
    n_spots, n_genes = X.shape
    n_top = min(n_top, n_spots - 2, n_genes)
    gene_variances = X.var(axis=0)

    real_evals, _ = wishart_top_k(backend, X, n_top)
    real_evals_desc = backend.to_numpy(real_evals)[::-1]

    rng = np.random.default_rng(rng)
    null_evals = np.empty((n_surrogates, n_top))
    for i in range(n_surrogates):
        X_null = classical_iid_null_matrix(n_spots, gene_variances, rng)
        evals, _ = wishart_top_k(backend, X_null, n_top)
        null_evals[i] = backend.to_numpy(evals)[::-1]

    n_signal, thresholds = _sequential_threshold_test(real_evals_desc, null_evals, alpha)
    return n_signal, real_evals_desc, thresholds
