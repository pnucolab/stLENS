from stLENS.backend import Backend
from stLENS.benchmark import run_benchmark_suite


def test_benchmark_suite_returns_positive_timings_for_small_sizes():
    backend = Backend(use_gpu=False)
    report = run_benchmark_suite(backend, sizes=(50, 200), n_genes=30, k_signal=5)
    assert set(report.keys()) == {50, 200}
    for size, stages in report.items():
        assert "spatial_graph" in stages
        assert "eigensolver" in stages
        assert stages["spatial_graph"]["seconds"] > 0
        assert stages["eigensolver"]["seconds"] > 0
