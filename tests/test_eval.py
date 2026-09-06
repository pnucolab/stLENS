import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from stLENS.eval import neighbor_confusion_matrix, optimize_resolution_for_all_celltypes_jaccard, pareto_frontier


class _FakeAdata:
    def __init__(self, obsp, obs, n_obs):
        self.obsp = obsp
        self.obs = obs
        self.n_obs = n_obs


def test_neighbor_confusion_matrix_perfect_when_neighbors_share_label():
    # two disconnected pairs, each pair sharing a label -> diagonal should be 1
    conn = csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float64))
    obs = pd.DataFrame({"cell_name": pd.Categorical(["A", "A", "B", "B"])})
    adata = _FakeAdata(obsp={"connectivities": conn}, obs=obs, n_obs=4)

    result = neighbor_confusion_matrix(adata, label_key="cell_name")
    np.testing.assert_allclose(np.diag(result.values), [1.0, 1.0])


def test_neighbor_confusion_matrix_handles_isolated_nodes():
    conn = csr_matrix((3, 3))  # no edges at all
    obs = pd.DataFrame({"cell_name": pd.Categorical(["A", "B", "A"])})
    adata = _FakeAdata(obsp={"connectivities": conn}, obs=obs, n_obs=3)
    result = neighbor_confusion_matrix(adata, label_key="cell_name")
    assert result.shape == (2, 2)
    assert np.isfinite(result.values).all()


def test_optimize_resolution_for_all_celltypes_jaccard_finds_perfect_match():
    obs = pd.DataFrame({
        "annotation": pd.Categorical(["A", "A", "B", "B"]),
        "leiden_0.5": pd.Categorical(["0", "0", "1", "1"]),
    })
    adata = _FakeAdata(obsp={}, obs=obs, n_obs=4)
    result = optimize_resolution_for_all_celltypes_jaccard(
        adata, average=None, annotation_key="annotation", res_range=[0.5]
    )
    assert set(result["cell_type"]) == {"A", "B"}
    np.testing.assert_allclose(result["best_jaccard"].to_numpy(), [1.0, 1.0])


def test_pareto_frontier_drops_dominated_points():
    records = [
        {"runtime_s": 1.0, "accuracy": 0.5},
        {"runtime_s": 2.0, "accuracy": 0.4},  # dominated: slower AND worse
        {"runtime_s": 3.0, "accuracy": 0.9},
        {"runtime_s": 5.0, "accuracy": 0.95},
    ]
    frontier = pareto_frontier(records)
    runtimes = [r["runtime_s"] for r in frontier]
    assert 2.0 not in runtimes
    assert 1.0 in runtimes and 3.0 in runtimes and 5.0 in runtimes
