"""Riemannian BCI covariance feature atoms.

Reimplementations of Riemannian geometry methods for EEG BCI, based on
algorithms from Barachant et al. (2012) and the pyRiemann library (BSD).
No pyRiemann dependency — all computations use numpy and scipy only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.signal
import scipy.linalg

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_autocorrelation_covariance_matrix,
    witness_cross_frequency_coherence_matrix,
    witness_tangent_space_projection,
    witness_riemannian_mean_spd,
)


def _is_symmetric(m: NDArray[np.float64], atol: float = 1e-10) -> bool:
    """Check whether a 2-D matrix is approximately symmetric."""
    return bool(m.ndim == 2 and m.shape[0] == m.shape[1] and np.allclose(m, m.T, atol=atol))


def _spd_power(m: NDArray[np.float64], exponent: float) -> NDArray[np.float64]:
    """Compute matrix power of an SPD matrix via eigendecomposition."""
    eigvals, eigvecs = scipy.linalg.eigh(m)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals ** exponent) @ eigvecs.T


@register_atom(witness_autocorrelation_covariance_matrix)
@icontract.require(lambda n_lags: n_lags >= 1, "n_lags must be >= 1")
@icontract.require(
    lambda data, n_lags: data.shape[1] > n_lags,
    "n_samples must be > n_lags",
)
@icontract.require(lambda data: data.ndim == 2, "data must be 2-D (n_channels, n_samples)")
@icontract.ensure(
    lambda result: _is_symmetric(result),
    "output covariance matrix must be symmetric",
)
def autocorrelation_covariance_matrix(
    data: NDArray[np.float64],
    n_lags: int,
) -> NDArray[np.float64]:
    """Build a Hankel autocorrelation covariance matrix from multichannel EEG.

    For each channel, the signal is stacked at K lags to form a Hankel-like
    augmented matrix.  The covariance of the stacked matrix yields an SPD
    matrix of size (n_channels * n_lags, n_channels * n_lags).

    Args:
        data: EEG data array of shape (n_channels, n_samples).
        n_lags: Number of autocorrelation lags (>= 1).

    Returns:
        SPD covariance matrix of shape (n_channels * n_lags, n_channels * n_lags).
    """
    n_channels, n_samples = data.shape
    usable = n_samples - n_lags + 1
    # Build the lagged / Hankel matrix: (n_channels * n_lags, usable)
    stacked = np.empty((n_channels * n_lags, usable), dtype=np.float64)
    for ch in range(n_channels):
        for lag in range(n_lags):
            stacked[ch * n_lags + lag, :] = data[ch, lag : lag + usable]

    # Covariance (bias-corrected)
    stacked_centered = stacked - stacked.mean(axis=1, keepdims=True)
    cov = (stacked_centered @ stacked_centered.T) / max(usable - 1, 1)
    # Ensure exact symmetry
    cov = (cov + cov.T) / 2.0
    return cov


@register_atom(witness_cross_frequency_coherence_matrix)
@icontract.require(lambda fs: fs > 0, "sampling frequency must be > 0")
@icontract.require(lambda data: data.ndim == 2 and data.shape[0] >= 2, "need >= 2 channels")
@icontract.require(lambda freq_bands: freq_bands.ndim == 2 and freq_bands.shape[1] == 2, "freq_bands must be (n_bands, 2)")
@icontract.require(lambda nperseg: nperseg >= 2, "nperseg must be >= 2")
@icontract.ensure(
    lambda result, data, freq_bands: result.shape == (freq_bands.shape[0], data.shape[0], data.shape[0]),
    "output shape must be (n_bands, n_channels, n_channels)",
)
def cross_frequency_coherence_matrix(
    data: NDArray[np.float64],
    fs: float,
    freq_bands: NDArray[np.float64],
    nperseg: int,
) -> NDArray[np.float64]:
    """Compute magnitude-squared coherence between all channel pairs at specified frequency bands.

    Uses :func:`scipy.signal.coherence` for each pair, then averages coherence
    within the requested frequency bands.

    Args:
        data: EEG data of shape (n_channels, n_samples).
        fs: Sampling frequency in Hz.
        freq_bands: Array of shape (n_bands, 2) with (low_hz, high_hz) per band.
        nperseg: Segment length for Welch coherence estimation.

    Returns:
        Coherence matrices of shape (n_bands, n_channels, n_channels).
    """
    n_channels = data.shape[0]
    n_bands = freq_bands.shape[0]
    result = np.zeros((n_bands, n_channels, n_channels), dtype=np.float64)

    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == j:
                for b in range(n_bands):
                    result[b, i, j] = 1.0
                continue
            freqs, coh = scipy.signal.coherence(
                data[i], data[j], fs=fs, nperseg=nperseg,
            )
            for b in range(n_bands):
                low, high = freq_bands[b]
                mask = (freqs >= low) & (freqs <= high)
                if mask.any():
                    result[b, i, j] = np.mean(coh[mask])
                    result[b, j, i] = result[b, i, j]

    return result


@register_atom(witness_tangent_space_projection)
@icontract.require(
    lambda matrices: matrices.ndim == 3 and matrices.shape[1] == matrices.shape[2] and matrices.shape[1] > 0,
    "matrices must be (n_matrices, p, p) with p > 0",
)
@icontract.require(
    lambda reference: reference.ndim == 2 and reference.shape[0] == reference.shape[1],
    "reference must be square",
)
@icontract.require(
    lambda matrices, reference: matrices.shape[1] == reference.shape[0],
    "matrices and reference must have the same p dimension",
)
@icontract.ensure(
    lambda result, matrices: result.shape == (matrices.shape[0], matrices.shape[1] * (matrices.shape[1] + 1) // 2),
    "output shape must be (n_matrices, p*(p+1)//2)",
)
def tangent_space_projection(
    matrices: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project SPD matrices to the tangent space at a reference point.

    Computes the matrix logarithm of the whitened matrices and extracts the
    upper-triangular elements (off-diagonal scaled by sqrt(2)) as tangent
    vectors.

    Args:
        matrices: SPD matrices of shape (n_matrices, p, p).
        reference: Reference SPD matrix of shape (p, p).

    Returns:
        Tangent vectors of shape (n_matrices, p*(p+1)//2).
    """
    n_matrices, p, _ = matrices.shape
    ref_sqrt_inv = _spd_power(reference, -0.5)
    vec_len = p * (p + 1) // 2
    tangent = np.empty((n_matrices, vec_len), dtype=np.float64)

    sqrt2 = np.sqrt(2.0)
    for k in range(n_matrices):
        whitened = ref_sqrt_inv @ matrices[k] @ ref_sqrt_inv
        # Ensure symmetry before logm
        whitened = (whitened + whitened.T) / 2.0
        log_whitened = scipy.linalg.logm(whitened).real
        # Extract upper triangle: diagonal as-is, off-diagonal * sqrt(2)
        idx = 0
        for i in range(p):
            for j in range(i, p):
                if i == j:
                    tangent[k, idx] = log_whitened[i, j]
                else:
                    tangent[k, idx] = log_whitened[i, j] * sqrt2
                idx += 1

    return tangent


@register_atom(witness_riemannian_mean_spd)
@icontract.require(
    lambda matrices: matrices.ndim == 3 and matrices.shape[0] >= 1 and matrices.shape[1] == matrices.shape[2],
    "matrices must be (n_matrices, p, p) with n_matrices >= 1",
)
@icontract.require(lambda max_iter: max_iter >= 1, "max_iter must be >= 1")
@icontract.require(lambda tol: tol > 0, "tol must be > 0")
@icontract.ensure(
    lambda result, matrices: result.shape == (matrices.shape[1], matrices.shape[2]),
    "output shape must be (p, p)",
)
def riemannian_mean_spd(
    matrices: NDArray[np.float64],
    max_iter: int = 50,
    tol: float = 1e-12,
) -> NDArray[np.float64]:
    """Iterative Karcher / Frechet mean on the SPD manifold.

    Initializes with the arithmetic mean, then iteratively refines via
    log-map averaging in the tangent space and exp-map back to the manifold.

    Args:
        matrices: SPD matrices of shape (n_matrices, p, p).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the Frobenius norm of the tangent mean.

    Returns:
        Riemannian mean SPD matrix of shape (p, p).
    """
    n_matrices, p, _ = matrices.shape

    # Initialize with arithmetic mean
    mean = np.mean(matrices, axis=0)
    mean = (mean + mean.T) / 2.0

    for _ in range(max_iter):
        mean_sqrt_inv = _spd_power(mean, -0.5)
        mean_sqrt = _spd_power(mean, 0.5)

        # Average log-maps in the tangent space at the current mean
        tangent_mean = np.zeros((p, p), dtype=np.float64)
        for k in range(n_matrices):
            whitened = mean_sqrt_inv @ matrices[k] @ mean_sqrt_inv
            whitened = (whitened + whitened.T) / 2.0
            tangent_mean += scipy.linalg.logm(whitened).real

        tangent_mean /= n_matrices

        if np.linalg.norm(tangent_mean, "fro") < tol:
            break

        # Exp-map back to the manifold
        mean = mean_sqrt @ scipy.linalg.expm(tangent_mean) @ mean_sqrt
        mean = (mean + mean.T) / 2.0

    return mean
