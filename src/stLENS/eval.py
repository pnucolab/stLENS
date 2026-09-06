"""Evaluation utilities.

`neighbor_confusion_matrix` and `optimize_resolution_for_all_celltypes_jaccard`
are factored out of stLENS-tutorials/spatial_notebook.ipynb so both the
speed-vs-accuracy harness and downstream validation reuse the exact same
evaluation code instead of duplicating it ad hoc. Plus a generic
speed-vs-accuracy scoring/Pareto-frontier helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def neighbor_confusion_matrix(adata, label_key="cell_name"):
    """For each ground-truth label, the fraction of a spot's kNN-graph
    neighbors that share each label -- a diagonal close to 1 means
    neighbors in embedding space tend to share the true cell type.

    `adata` needs `.obsp["connectivities"]` (a neighbor graph, e.g. from
    `sc.pp.neighbors`), `.obs[label_key]` (categorical ground truth), and
    `.n_obs`. Ported from stLENS-tutorials/spatial_notebook.ipynb.
    """
    conn = adata.obsp["connectivities"].tolil()
    labels = adata.obs[label_key].astype("category")
    label_codes = labels.cat.codes
    label_names = labels.cat.categories
    n_labels = len(label_names)

    mat = np.zeros((n_labels, n_labels))
    for i in range(adata.n_obs):
        src = label_codes[i]
        neighbors = [j for j in conn.rows[i] if j != i]
        if not neighbors:
            continue
        for tgt in label_codes[neighbors]:
            mat[src, tgt] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums
    return pd.DataFrame(mat, index=label_names, columns=label_names)


def optimize_resolution_for_all_celltypes_jaccard(adata, average, annotation_key="annotation", res_range=None):
    """For each ground-truth cell type, the best Jaccard score achievable
    by any Leiden cluster at any tested resolution.

    `adata.obs` needs the categorical `annotation_key` column and one
    categorical `leiden_{res:.1f}` column per resolution in `res_range`
    (e.g. from running `sc.tl.leiden` at several resolutions). Ported from
    stLENS-tutorials/spatial_notebook.ipynb.
    """
    from sklearn.metrics import jaccard_score

    if res_range is None:
        res_range = np.arange(0.5, 5.1, 0.5)

    summary = []
    cell_types = adata.obs[annotation_key].cat.categories
    for cell_type in cell_types:
        gt_mask = (adata.obs[annotation_key] == cell_type).to_numpy()
        if gt_mask.sum() == 0:
            continue

        for res in res_range:
            leiden_key = f"leiden_{res:.1f}"
            if leiden_key not in adata.obs.columns:
                continue

            best_jaccard, best_cluster = -1.0, None
            for cluster in adata.obs[leiden_key].cat.categories:
                pred_mask = (adata.obs[leiden_key] == cluster).to_numpy()
                score = jaccard_score(gt_mask, pred_mask, average=average)
                score = score[1] if hasattr(score, "__len__") else score
                if score > best_jaccard:
                    best_jaccard, best_cluster = score, cluster

            summary.append({
                "cell_type": cell_type,
                "resolution": res,
                "best_jaccard": best_jaccard,
                "best_cluster": best_cluster,
            })

    return pd.DataFrame(summary)


def pareto_frontier(records, time_key="runtime_s", accuracy_key="accuracy"):
    """Return the subset of `records` (list of dicts) that are
    Pareto-optimal for (lower runtime, higher accuracy), so a
    speed-vs-accuracy sweep can be reduced to just its useful configs."""
    records_sorted = sorted(records, key=lambda r: r[time_key])
    frontier = []
    best_acc = -np.inf
    for r in records_sorted:
        if r[accuracy_key] > best_acc:
            frontier.append(r)
            best_acc = r[accuracy_key]
    return frontier
