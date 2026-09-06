"""Top-level spatial pipeline, added in this version of stLENS.

Combines the backend, truncated eigensolver, spatially-aware
normalization, and the spatial random-matrix null model into one entry
point that returns BOTH a plain (non-spatial) embedding and a spatially-
regularized embedding side by side, for backward-compatible comparison
with stLENS's original non-spatial output.

The number of components in the spatial embedding is chosen by
`spatial_mp_threshold` (see `spatial_null.py`) -- a spatially-corrected
generalization of the classical Marchenko-Pastur/Tracy-Widom RMT
threshold that does not systematically over-count spatially-correlated
technical noise as signal, which is this pipeline's core contribution,
not just a speed optimization. The spatial-graph SRT + Moran's I check
(`spatial_srt`) is kept as an independent secondary diagnostic of which
specific components look spatially coherent.
"""

from __future__ import annotations

import numpy as np

from .backend import get_backend
from .eigensolver import wishart_top_k
from .normalize import spatial_normalize
from .spatial_null import spatial_mp_threshold
from .srt import spatial_srt


def run_spatial_pipeline(X, coords, k_candidate=30, n_perturb=20, n_surrogates=150, alpha=0.05,
                          backend=None, rng=None):
    """Run the spatial pipeline end-to-end on a dense/sparse spot x gene
    matrix `X` and spot coordinates `coords`.

    Returns a dict with:
      - "X_pca": plain PCA embedding (z-score normalize + truncated
        eigensolver only, no spatial correction) -- backward-compatible
        with stLENS's original non-spatial output.
      - "X_pca_spatial": spatially-regularized embedding, using
        spatially-aware normalization and the spatial random-matrix null
        model to pick the number of components.
      - "n_signal_pcs": the number of components confirmed by
        `spatial_mp_threshold` -- the spatially-corrected RMT threshold.
      - "moran_scores": Moran's I of the top `k_candidate` components
        against the spatial graph (secondary diagnostic from
        `spatial_srt`; does not affect `n_signal_pcs`).
      - "spatial_null_thresholds": the empirical spatial-null threshold
        used at each rank, for inspection/plotting against the real
        eigenvalues.
    """
    backend = backend or get_backend()
    k_candidate = min(k_candidate, X.shape[0] - 2, X.shape[1])

    X_dense = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X, dtype=np.float64)

    # Plain (non-spatial) path.
    mean = X_dense.mean(axis=0)
    std = X_dense.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_plain = (X_dense - mean) / std
    evals_plain, evecs_plain = wishart_top_k(backend, X_plain, k_candidate)
    X_pca = backend.to_numpy(evecs_plain) * np.sqrt(np.clip(backend.to_numpy(evals_plain), 0, None))

    # Spatially-regularized path.
    X_spatial = spatial_normalize(X_dense, coords)

    n_signal_pcs, _, null_thresholds = spatial_mp_threshold(
        backend, X_spatial, coords, alpha=alpha, n_surrogates=n_surrogates, n_top=k_candidate, rng=rng,
    )
    _, moran_scores = spatial_srt(
        backend, X_spatial, coords, k_signal=k_candidate, n_perturb=n_perturb, rng=rng,
    )

    n_out = max(n_signal_pcs, 1)
    evals_s, evecs_s = wishart_top_k(backend, X_spatial, n_out)
    X_pca_spatial = backend.to_numpy(evecs_s) * np.sqrt(np.clip(backend.to_numpy(evals_s), 0, None))

    return {
        "X_pca": X_pca,
        "X_pca_spatial": X_pca_spatial,
        "n_signal_pcs": n_signal_pcs,
        "moran_scores": moran_scores,
        "spatial_null_thresholds": null_thresholds,
    }
