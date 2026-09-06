"""Truncated/randomized eigensolver for stLENS.

stLENS's sparsity search and SRT loop only ever consume the top-k
eigenpairs of a (Wishart) matrix, but the reference implementation always
computes a *full* `eigh`: O(d^3) instead of the O(d^2 k) a truncated
solver needs. On large spatial transcriptomics matrices (d = thousands to
tens of thousands of genes/features, repeated ~20x per SRT run), this is
the single biggest, most general speedup available before any
spatial-specific work.
"""

from __future__ import annotations


def top_k_eigh(backend, a, k, which="LA"):
    """Top-k eigenpairs of a symmetric/Hermitian matrix `a`.

    Returns `(eigenvalues, eigenvectors)` sorted ASCENDING by eigenvalue,
    eigenvectors as columns — the same convention as `numpy`/`cupy`'s
    `linalg.eigh`, so this is a drop-in replacement for the pattern
    `evals, evecs = xp.linalg.eigh(a); evals[-k:], evecs[:, -k:]`.

    Falls back to a full `eigh` when `k` is not meaningfully smaller than
    the matrix size: truncated iterative solvers require `k <= n - 2` and
    stop being a speed win as `k` approaches `n`.
    """
    xp = backend.xp
    n = a.shape[0]
    k = min(k, n)

    if k <= 0:
        evals, evecs = xp.linalg.eigh(a)
        return evals[:0], evecs[:, :0]

    if k >= n - 1:
        evals, evecs = xp.linalg.eigh(a)
        return evals[-k:], evecs[:, -k:]

    if backend.is_gpu:
        import cupyx.scipy.sparse.linalg as gsla

        evals, evecs = gsla.eigsh(a, k=k, which=which)
    else:
        import scipy.sparse.linalg as sla

        a_host = backend.to_numpy(a)
        evals, evecs = sla.eigsh(a_host, k=k, which=which)
        evals = xp.asarray(evals)
        evecs = xp.asarray(evecs)

    order = xp.argsort(evals)
    return evals[order], evecs[:, order]


def wishart_top_k(backend, X, k, which="LA"):
    """Top-k eigenpairs of the (implicit) spot x spot Wishart matrix `X @
    X.T`, computed via whichever of `X @ X.T` (spot x spot) or `X.T @ X`
    (gene x gene) is smaller.

    This matters a lot for large spatial transcriptomics data: forming the
    spot x spot Gram matrix directly for e.g. 1M spots would need an 8 TB
    dense matrix, even though there are usually only a few hundred to a
    few thousand genes. Decomposing the smaller Gram matrix and projecting
    back (the standard PCA-via-covariance trick: `v_spot = X @ v_gene /
    sqrt(eigenvalue)`) gives numerically equivalent results at a cost that
    no longer explodes with spot count.

    Always returns eigenvectors in SPOT space (shape `(n_spots, k)`), so
    callers (PCA embedding, Moran's I over spots) don't need to know which
    side was actually decomposed.
    """
    xp = backend.xp
    X = backend.asarray(X)
    n_spots, n_genes = X.shape

    if n_spots <= n_genes:
        gram = X @ X.T
        return top_k_eigh(backend, gram, k, which=which)

    gram = X.T @ X
    evals, evecs_gene = top_k_eigh(backend, gram, k, which=which)  # (n_genes, k)

    safe_evals = xp.where(evals > 1e-12, evals, 1.0)
    evecs_spot = (X @ evecs_gene) / xp.sqrt(safe_evals)[None, :]
    return evals, evecs_spot
