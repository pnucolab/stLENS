import numpy as np

from stLENS.variogram import calibrate_hyperparameters, estimate_autocorrelation_range, subsampled_variogram


def _smooth_gaussian_field(side, length_scale, seed):
    """A grid field smoothed by a Gaussian kernel of the given sigma, via a
    simple FFT convolution -- gives a field with a known-ish correlation
    length for sanity-checking the variogram estimator."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((side, side))

    xs, ys = np.meshgrid(np.arange(-side // 2, side // 2), np.arange(-side // 2, side // 2))
    kernel = np.exp(-(xs ** 2 + ys ** 2) / (2 * length_scale ** 2))
    kernel /= kernel.sum()

    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.fft.fft2(np.fft.ifftshift(kernel))))

    xs_c, ys_c = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([xs_c.ravel(), ys_c.ravel()], axis=1).astype(np.float64)
    return coords, field.ravel()


def test_variogram_range_scales_with_true_length_scale():
    coords_short, values_short = _smooth_gaussian_field(side=60, length_scale=2.0, seed=0)
    coords_long, values_long = _smooth_gaussian_field(side=60, length_scale=10.0, seed=0)

    bc_short, g_short = subsampled_variogram(coords_short, values_short, n_pairs=15000, rng=1)
    bc_long, g_long = subsampled_variogram(coords_long, values_long, n_pairs=15000, rng=1)

    range_short = estimate_autocorrelation_range(bc_short, g_short)
    range_long = estimate_autocorrelation_range(bc_long, g_long)

    assert range_long > range_short


def test_calibrate_hyperparameters_returns_positive_values():
    params = calibrate_hyperparameters(7.5)
    assert params["normalization_knn_radius"] > 0
    assert params["srt_block_radius"] > 0
    assert params["coreset_hex_radius"] > 0
    assert params["coreset_hex_radius"] < params["normalization_knn_radius"]


def test_calibrate_hyperparameters_handles_zero_or_negative_input():
    params = calibrate_hyperparameters(0.0)
    assert params["normalization_knn_radius"] > 0
