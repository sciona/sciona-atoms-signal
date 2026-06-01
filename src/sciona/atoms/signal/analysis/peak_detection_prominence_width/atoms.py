from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_find_local_maxima,
    witness_compute_peak_prominences,
    witness_compute_peak_widths,
)

@register_atom(witness_find_local_maxima, name="find_local_maxima")
@icontract.require(lambda signal: signal.ndim == 1, "Precondition failed: signal.ndim == 1")
@icontract.ensure(lambda result, signal: result.ndim == 1, "Postcondition failed: result.ndim == 1")
@icontract.ensure(lambda result, signal: np.all(result >= 0), "Postcondition failed: np.all(result >= 0)")
def find_local_maxima(signal: NDArray[np.float64]) -> NDArray[np.int64]:
    """Identifies indices of all local maximum peaks in a 1D signal.

    Args:
        signal: NDArray[np.float64]

    Returns:
        result: NDArray[np.int64]
    """
    import scipy.signal
    return scipy.signal.find_peaks(signal=signal) # type: ignore

@register_atom(witness_compute_peak_prominences, name="compute_peak_prominences")
@icontract.require(lambda signal, peaks: signal.ndim == 1, "Precondition failed: signal.ndim == 1")
@icontract.require(lambda signal, peaks: peaks.ndim == 1, "Precondition failed: peaks.ndim == 1")
@icontract.ensure(lambda result, signal, peaks: len(result) == 3, "Postcondition failed: len(result) == 3")
@icontract.ensure(lambda result, signal, peaks: result[0].shape == peaks.shape, "Postcondition failed: result[0].shape == peaks.shape")
def compute_peak_prominences(signal: NDArray[np.float64], peaks: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Calculates the prominence of each detected peak.

    Args:
        signal: NDArray[np.float64]
        peaks: NDArray[np.int64]

    Returns:
        result: tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]
    """
    import scipy.signal
    return scipy.signal.peak_prominences(signal=signal, peaks=peaks) # type: ignore

@register_atom(witness_compute_peak_widths, name="compute_peak_widths")
@icontract.require(lambda signal, peaks, prominences, rel_height: signal.ndim == 1, "Precondition failed: signal.ndim == 1")
@icontract.require(lambda signal, peaks, prominences, rel_height: peaks.ndim == 1, "Precondition failed: peaks.ndim == 1")
@icontract.ensure(lambda result, signal, peaks, prominences, rel_height: len(result) == 4, "Postcondition failed: len(result) == 4")
def compute_peak_widths(signal: NDArray[np.float64], peaks: NDArray[np.int64], prominences: NDArray[np.float64], rel_height: float = None) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculates the width of each peak at a relative height.

    Args:
        signal: NDArray[np.float64]
        peaks: NDArray[np.int64]
        prominences: NDArray[np.float64]
        rel_height: float

    Returns:
        result: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    """
    import scipy.signal
    return scipy.signal.peak_widths(signal=signal, peaks=peaks, prominences=prominences, rel_height=rel_height) # type: ignore

