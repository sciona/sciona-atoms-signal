from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_frame_periodograms,
    witness_average_periodograms,
)

@register_atom(witness_compute_frame_periodograms, name="compute_frame_periodograms")
@icontract.require(lambda stft_matrix, window_norm: stft_matrix.ndim == 2, "Precondition failed: stft_matrix.ndim == 2")
@icontract.require(lambda stft_matrix, window_norm: window_norm > 0, "Precondition failed: window_norm > 0")
@icontract.ensure(lambda result, stft_matrix, window_norm: result.ndim == 2, "Postcondition failed: result.ndim == 2")
@icontract.ensure(lambda result, stft_matrix, window_norm: result.shape == stft_matrix.shape, "Postcondition failed: result.shape == stft_matrix.shape")
@icontract.ensure(lambda result, stft_matrix, window_norm: np.all(result >= 0), "Postcondition failed: np.all(result >= 0)")
def compute_frame_periodograms(stft_matrix: NDArray[np.complex128], window_norm: float) -> NDArray[np.float64]:
    """Computes the squared magnitude of the windowed FFT for each frame to obtain the periodogram.

    This corresponds to extracting the magnitude spectrum from complex Fourier/FFT output,
    matching the "Magnitude Spectrum" / "extract_magnitude" step in benchmarks.

    Args:
        stft_matrix: NDArray[np.complex128]
        window_norm: Energy norm of the window function

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.welch(stft_matrix=stft_matrix, window_norm=window_norm) # type: ignore

@register_atom(witness_average_periodograms, name="average_periodograms")
@icontract.require(lambda periodograms: periodograms.ndim == 2, "Precondition failed: periodograms.ndim == 2")
@icontract.ensure(lambda result, periodograms: result.ndim == 1, "Postcondition failed: result.ndim == 1")
@icontract.ensure(lambda result, periodograms: result.shape[0] == periodograms.shape[0], "Postcondition failed: result.shape[0] == periodograms.shape[0]")
def average_periodograms(periodograms: NDArray[np.float64]) -> NDArray[np.float64]:
    """Averages periodograms across all frames to reduce spectral variance.

    Args:
        periodograms: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import numpy
    return numpy.mean(periodograms=periodograms) # type: ignore

