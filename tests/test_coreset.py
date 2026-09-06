import numpy as np
from scipy.sparse import csr_matrix

from stLENS.coreset import (
    aggregate_counts,
    hex_bin,
    knn_pseudobulk,
    recommend_default_group_size,
    sweep_compression_ratios,
    upsample_distance_weighted,
    upsample_nearest,
)


def test_hex_bin_groups_tight_clusters_together_and_far_points_apart():
    cluster_a = np.array([0.0, 0.0]) + 0.1 * np.random.default_rng(0).standard_normal((20, 2))
    cluster_b = np.array([100.0, 100.0]) + 0.1 * np.random.default_rng(1).standard_normal((20, 2))
    coords = np.vstack([cluster_a, cluster_b])

    groups = hex_bin(coords, hex_radius=5.0)
    assert len(np.unique(groups[:20])) == 1
    assert len(np.unique(groups[20:])) == 1
    assert groups[0] != groups[20]


def test_knn_pseudobulk_assigns_every_spot_exactly_once():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 50, size=(237, 2))
    groups = knn_pseudobulk(coords, group_size=10)
    assert groups.shape == (237,)
    assert (groups >= 0).all()
    _, counts = np.unique(groups, return_counts=True)
    assert counts.sum() == 237


def test_knn_pseudobulk_group_sizes_mostly_hit_target():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 50, size=(500, 2))
    group_size = 20
    groups = knn_pseudobulk(coords, group_size=group_size)
    _, counts = np.unique(groups, return_counts=True)

    assert counts.max() <= group_size
    assert np.sum(counts == group_size) >= 0.8 * len(counts)


def test_aggregate_counts_matches_manual_sum():
    X = csr_matrix(np.arange(24).reshape(8, 3).astype(np.float64))
    group_ids = np.array([0, 0, 1, 1, 2, 2, 2, 2])

    result = aggregate_counts(X, group_ids, chunk_size=3)
    result_dense = np.asarray(result.todense())

    expected = np.zeros((3, 3))
    X_dense = np.asarray(X.todense())
    for g in range(3):
        expected[g] = X_dense[group_ids == g].sum(axis=0)

    np.testing.assert_allclose(result_dense, expected)


def test_upsample_nearest_matches_group_value():
    values = np.array([10.0, 20.0, 30.0])
    group_ids = np.array([0, 0, 1, 2, 1])
    result = upsample_nearest(values, group_ids)
    np.testing.assert_array_equal(result, [10.0, 10.0, 20.0, 30.0, 20.0])


def test_upsample_distance_weighted_interpolates_between_neighbors():
    group_coords = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([0.0, 10.0])
    coords = np.array([[5.0, 0.0]])  # exactly midway
    result = upsample_distance_weighted(values, coords, group_coords, k=2)
    np.testing.assert_allclose(result, [5.0], atol=1e-6)


def test_sweep_compression_ratios_reports_increasing_ratio_for_larger_groups():
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 50, size=(300, 2))

    def fake_evaluator(group_ids):
        return {"accuracy": 1.0 / (1 + group_ids.max())}

    records = sweep_compression_ratios(coords, [5, 20, 50], fake_evaluator)
    ratios = [r["compression_ratio"] for r in records]
    assert ratios == sorted(ratios)


def test_recommend_default_group_size_increases_with_dataset_size():
    small = recommend_default_group_size(1_000)
    medium = recommend_default_group_size(20_000)
    large = recommend_default_group_size(200_000)
    huge = recommend_default_group_size(2_000_000)
    assert small <= medium <= large <= huge
    assert small == 1
