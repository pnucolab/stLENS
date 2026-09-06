from . import stLENS as _
stLENS = _.stLENS

from .backend import Backend, get_backend
from .eigensolver import top_k_eigh, wishart_top_k
from .spatial_graph import build_spatial_knn_graph, to_similarity_weights, row_normalize
from .coreset import (
    hex_bin, knn_pseudobulk, aggregate_counts, upsample_nearest, upsample_distance_weighted,
    sweep_compression_ratios, recommend_default_group_size,
)
from .normalize import local_background, spatial_normalize
from .srt import morans_i, morans_i_batch, spatial_block_permute, spatial_srt
from .variogram import subsampled_variogram, estimate_autocorrelation_range, calibrate_hyperparameters
from .synthetic import generate_spatial_dataset, generate_spatial_dataset_with_spatial_noise
from .spatial_null import (
    spatial_smoothing_operator, generate_spatial_null_matrix, classical_iid_null_matrix,
    spatial_mp_threshold, classical_iid_threshold,
)
from .pipeline import run_spatial_pipeline

__version__ = "0.3.0"
__all__ = [
    "stLENS",
    "Backend", "get_backend",
    "top_k_eigh", "wishart_top_k",
    "build_spatial_knn_graph", "to_similarity_weights", "row_normalize",
    "hex_bin", "knn_pseudobulk", "aggregate_counts", "upsample_nearest", "upsample_distance_weighted",
    "sweep_compression_ratios", "recommend_default_group_size",
    "local_background", "spatial_normalize",
    "morans_i", "morans_i_batch", "spatial_block_permute", "spatial_srt",
    "subsampled_variogram", "estimate_autocorrelation_range", "calibrate_hyperparameters",
    "generate_spatial_dataset", "generate_spatial_dataset_with_spatial_noise",
    "spatial_smoothing_operator", "generate_spatial_null_matrix", "classical_iid_null_matrix",
    "spatial_mp_threshold", "classical_iid_threshold",
    "run_spatial_pipeline",
    "__version__",
]
