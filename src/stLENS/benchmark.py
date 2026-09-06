"""Multi-scale wall-clock/memory benchmark harness.

Times each stLENS stage across increasing synthetic spot counts. Real
Visium HD / STOmics runs plug into the same `time_stage` helper once those
datasets are loaded (see roadmap "Head-to-head speed/accuracy comparison"
task) -- this module provides the reusable timing/instrumentation
machinery itself, exercised here on synthetic data of arbitrary size.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager


@contextmanager
def time_stage(name, results: dict):
    """Record wall-clock seconds and peak memory (bytes) for one stage
    into `results[name]`."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = {"seconds": elapsed, "peak_memory_bytes": peak}


def run_benchmark_suite(backend, sizes=(1000, 5000, 20000), n_genes=200, k_signal=10, seed=0):
    """Time spatial-graph construction and the truncated Wishart
    eigensolver across increasing synthetic spot counts.

    Uses `wishart_top_k`, which decomposes whichever of `X @ X.T` / `X.T @
    X` is smaller, so this stays memory-safe even at very large spot
    counts (n_genes stays fixed, so the eigendecomposition cost does not
    grow with n_spots -- only the O(n_spots * n_genes) matmuls do).
    """
    from .eigensolver import wishart_top_k
    from .spatial_graph import build_spatial_knn_graph
    from .synthetic import generate_spatial_dataset

    report = {}
    for n_spots in sizes:
        X, coords, _, _ = generate_spatial_dataset(n_spots=n_spots, n_genes=n_genes, seed=seed)
        stage_results = {}

        with time_stage("spatial_graph", stage_results):
            build_spatial_knn_graph(coords, k=15)

        with time_stage("eigensolver", stage_results):
            wishart_top_k(backend, X, min(k_signal, n_spots - 2, n_genes))

        report[n_spots] = stage_results

    return report
