"""Ghost witnesses for the Riemannian BCI covariance feature atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_autocorrelation_covariance_matrix(
    data: AbstractArray,
    n_lags: int,
) -> AbstractArray:
    """Autocorrelation covariance produces a square SPD matrix of size (n_channels * n_lags)."""
    n_channels = data.shape[0]
    dim = n_channels * n_lags
    return AbstractArray(shape=(dim, dim), dtype="float64")


def witness_cross_frequency_coherence_matrix(
    data: AbstractArray,
    fs: float,
    freq_bands: AbstractArray,
    nperseg: int,
) -> AbstractArray:
    """Coherence matrix has shape (n_bands, n_channels, n_channels)."""
    n_channels = data.shape[0]
    n_bands = freq_bands.shape[0]
    return AbstractArray(shape=(n_bands, n_channels, n_channels), dtype="float64")


def witness_tangent_space_projection(
    matrices: AbstractArray,
    reference: AbstractArray,
) -> AbstractArray:
    """Tangent vectors have shape (n_matrices, p*(p+1)//2)."""
    n_matrices = matrices.shape[0]
    p = matrices.shape[1]
    vec_len = p * (p + 1) // 2
    return AbstractArray(shape=(n_matrices, vec_len), dtype="float64")


def witness_riemannian_mean_spd(
    matrices: AbstractArray,
    max_iter: int,
    tol: float,
) -> AbstractArray:
    """Riemannian mean has shape (p, p)."""
    p = matrices.shape[1]
    return AbstractArray(shape=(p, p), dtype="float64")
