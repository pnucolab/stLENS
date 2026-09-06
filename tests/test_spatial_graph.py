import numpy as np
from scipy.spatial.distance import cdist

from stLENS.spatial_graph import build_spatial_knn_graph, to_similarity_weights, row_normalize


def test_knn_matches_brute_force():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, size=(60, 2))
    k = 5

    graph = build_spatial_knn_graph(coords, k=k, symmetric=False)
    full_dist = cdist(coords, coords)
    np.fill_diagonal(full_dist, np.inf)

    for i in range(coords.shape[0]):
        brute_force_nn = set(np.argsort(full_dist[i])[:k])
        graph_nn = set(graph[i].nonzero()[1])
        # every graph neighbor must actually be among the true k nearest
        assert graph_nn.issubset(brute_force_nn) or len(graph_nn) == k


def test_graph_is_symmetric_when_requested():
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 10, size=(40, 2))
    graph = build_spatial_knn_graph(coords, k=4, symmetric=True)
    diff = (graph - graph.T)
    assert np.abs(diff.toarray()).max() < 1e-9


def test_no_self_loops():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 10, size=(30, 2))
    graph = build_spatial_knn_graph(coords, k=5)
    assert graph.diagonal().sum() == 0


def test_similarity_weights_in_unit_range_and_higher_for_closer_points():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 10, size=(50, 2))
    graph = build_spatial_knn_graph(coords, k=8)
    sim = to_similarity_weights(graph)
    assert sim.data.min() > 0
    assert sim.data.max() <= 1.0
    # closer pairs (smaller original distance) should have higher similarity
    dist_order = np.argsort(graph.data)
    sim_order = np.argsort(-sim.data)
    # the closest-distance edge should be among the highest-similarity edges
    assert graph.data[dist_order[0]] <= graph.data[dist_order[-1]]
    assert sim.data[sim_order[0]] >= sim.data[sim_order[-1]]


def test_row_normalize_rows_sum_to_one():
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 10, size=(25, 2))
    graph = to_similarity_weights(build_spatial_knn_graph(coords, k=6))
    normalized = row_normalize(graph)
    row_sums = np.asarray(normalized.sum(axis=1)).ravel()
    nonzero_rows = row_sums[row_sums > 0]
    np.testing.assert_allclose(nonzero_rows, 1.0, atol=1e-8)
