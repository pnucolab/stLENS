import numpy as np

from stLENS.backend import Backend
from stLENS.spatial_graph import build_spatial_knn_graph
from stLENS.srt import morans_i, morans_i_batch, spatial_block_permute, spatial_srt
from stLENS.synthetic import generate_spatial_dataset


def _grid_coords(n):
    side = int(np.sqrt(n))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def test_morans_i_high_for_smooth_spatial_signal():
    coords = _grid_coords(400)
    graph = build_spatial_knn_graph(coords, k=8)
    smooth_signal = coords[:, 0]  # perfectly spatially structured
    assert morans_i(smooth_signal, graph) > 0.5


def test_morans_i_near_zero_for_random_noise():
    coords = _grid_coords(400)
    graph = build_spatial_knn_graph(coords, k=8)
    noise = np.random.default_rng(0).standard_normal(coords.shape[0])
    assert abs(morans_i(noise, graph)) < 0.15


def test_morans_i_batch_matches_single_calls():
    coords = _grid_coords(200)
    graph = build_spatial_knn_graph(coords, k=6)
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((coords.shape[0], 4))
    vectors[:, 0] = coords[:, 0]  # one spatially structured column

    batch_scores = morans_i_batch(vectors, graph)
    single_scores = [morans_i(vectors[:, c], graph) for c in range(vectors.shape[1])]
    np.testing.assert_allclose(batch_scores, single_scores, atol=1e-8)


def test_spatial_block_permute_preserves_value_multiset():
    coords = _grid_coords(100)
    X = np.random.default_rng(2).standard_normal((coords.shape[0], 5))
    X_perm = spatial_block_permute(X, coords, block_size=10, rng=3)
    np.testing.assert_allclose(np.sort(X.ravel()), np.sort(X_perm.ravel()))
    assert not np.allclose(X, X_perm)


def test_spatial_srt_recovers_roughly_true_signal_count():
    X, coords, true_k, _ = generate_spatial_dataset(
        n_spots=900, n_genes=100, n_signal_components=4, spatial_length_scale=8.0,
        noise_level=0.5, seed=7,
    )
    backend = Backend(use_gpu=False)
    n_confirmed, scores = spatial_srt(
        backend, X, coords, k_signal=10, n_perturb=8, block_size=30, rng=0
    )
    assert 1 <= n_confirmed <= 8
    # the confirmed count shouldn't wildly overshoot the true count
    assert n_confirmed <= true_k + 4
