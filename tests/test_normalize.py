import numpy as np

from stLENS.normalize import local_background, spatial_normalize


def _make_sparse_count_dataset(n=400, n_genes=15, mean_counts_per_gene=0.4, seed=0):
    """Low-count synthetic data where the TRUE expected total count per spot
    follows a smooth spatial gradient, but each spot's OWN observed total is
    a noisy Poisson draw around it -- the classic case where borrowing
    strength from spatial neighbors gives a better estimate of the true
    local scale than the spot's own (noisy) count.
    """
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    smooth_rate_per_gene = mean_counts_per_gene * (0.3 + coords[:, 0] / coords[:, 0].max())
    true_rate = n_genes * smooth_rate_per_gene  # expected TOTAL count per spot

    gene_profile = rng.gamma(shape=2.0, scale=1.0, size=n_genes)
    gene_profile /= gene_profile.mean()
    lam = smooth_rate_per_gene[:, None] * gene_profile[None, :]
    X = rng.poisson(lam).astype(np.float64)

    return X, coords, true_rate


def test_local_background_estimates_true_scale_better_than_raw_totals():
    X, coords, true_rate = _make_sparse_count_dataset()
    totals = X.sum(axis=1)
    background = local_background(X, coords, k=10)

    totals_mse = np.mean((totals - true_rate) ** 2)
    background_mse = np.mean((background - true_rate) ** 2)
    # neighbor-averaging denoises the low-count per-spot total towards the
    # smooth true rate; a single spot's own noisy total cannot do this.
    assert background_mse < totals_mse


def test_spatial_normalize_output_shape_and_no_nans():
    X, coords, _ = _make_sparse_count_dataset(n=100, n_genes=20)
    out = spatial_normalize(X, coords, k=8)
    assert out.shape == (100, 20)
    assert np.isfinite(out).all()


def test_spatial_normalize_isolated_spot_falls_back_to_own_total():
    # a single far-away spot has no meaningful neighbors within the graph's
    # k, so local_background should not blow up or divide by zero for it.
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 10, size=(30, 2))
    coords = np.vstack([coords, [[10_000.0, 10_000.0]]])  # outlier spot
    X = rng.poisson(3.0, size=(31, 10)).astype(np.float64)

    background = local_background(X, coords, k=5)
    assert np.isfinite(background).all()
    assert background[-1] > 0
