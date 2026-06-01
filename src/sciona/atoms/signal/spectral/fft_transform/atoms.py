"""FFT spectral transform atoms wrapping NumPy and SciPy operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_apply_spectral_window,
    witness_optimize_fft_length,
    witness_compute_forward_rfft,
)


@register_atom(witness_apply_spectral_window)  # type: ignore[untyped-decorator]
@icontract.require(lambda signal: signal.ndim == 1, "Input signal must be a 1D array")
@icontract.require(lambda signal: np.all(np.isfinite(signal)), "Input signal must contain only finite values")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "Output must have the same shape as input")
def apply_spectral_window(
    signal: NDArray[np.float64],
    window_type: str = "hann",
) -> NDArray[np.float64]:
    """Applies a specified window to the input signal to reduce spectral leakage.

    Parameters
    ----------
    signal : NDArray[np.float64]
        1D finite real signal array.
    window_type : str, optional
        A valid SciPy window identifier (e.g. "hann", "hamming", "boxcar").
        Defaults to "hann".

    Returns
    -------
    NDArray[np.float64]
        The windowed signal of the same shape.
    """
    import scipy.signal.windows
    window = scipy.signal.windows.get_window(window_type, len(signal))
    return signal * window  # type: ignore[no-any-return]


@register_atom(witness_optimize_fft_length)  # type: ignore[untyped-decorator]
@icontract.require(lambda target_len: target_len > 0, "Target length must be a positive integer")
@icontract.ensure(lambda result, target_len: result >= target_len, "Optimal length must be >= target length")
def optimize_fft_length(target_len: int) -> int:
    """Finds the next highly composite size (powers of 2, 3, 5) for efficient FFT execution.

    Parameters
    ----------
    target_len : int
        A positive integer representing the minimum desired FFT length.

    Returns
    -------
    int
        The next optimal highly composite FFT size.
    """
    import scipy.fft
    return int(scipy.fft.next_fast_len(target_len))


@register_atom(witness_compute_forward_rfft)  # type: ignore[untyped-decorator]
@icontract.require(lambda signal: signal.ndim == 1, "Input signal must be a 1D array")
@icontract.require(lambda signal, n: n >= len(signal), "FFT length n must be >= signal length")
@icontract.ensure(lambda result, n: result.shape[0] == n // 2 + 1, "Output shape must be n // 2 + 1")
def compute_forward_rfft(
    signal: NDArray[np.float64],
    n: int,
) -> NDArray[np.complex128]:
    """Computes the one-dimensional discrete Fourier Transform for real input (RFFT).

    Parameters
    ----------
    signal : NDArray[np.float64]
        1D real signal array to transform.
    n : int
        The target FFT length (greater than or equal to signal length).

    Returns
    -------
    NDArray[np.complex128]
        The complex spectrum of length n // 2 + 1.
    """
    import scipy.fft
    return scipy.fft.rfft(signal, n=n)  # type: ignore[no-any-return]
