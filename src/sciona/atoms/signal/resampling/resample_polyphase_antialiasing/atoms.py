from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_rational_approximation,
    witness_design_resampling_filter,
    witness_apply_polyphase_filtering,
)

@register_atom(witness_compute_rational_approximation, name="compute_rational_approximation")
@icontract.require(lambda up, down, max_len: up > 0, "Precondition failed: up > 0")
@icontract.require(lambda up, down, max_len: down > 0, "Precondition failed: down > 0")
@icontract.ensure(lambda result, up, down, max_len: result[0] > 0, "Postcondition failed: result[0] > 0")
@icontract.ensure(lambda result, up, down, max_len: result[1] > 0, "Postcondition failed: result[1] > 0")
def compute_rational_approximation(up: float, down: float, max_len: int = None) -> int:
    """Approximates a floating-point resampling ratio with a rational fraction P/Q.

    Args:
        up: float
        down: float
        max_len: int

    Returns:
        result: tuple[int, int]
    """
    import scipy.signal
    return scipy.signal.resample_poly(up=up, down=down, max_len=max_len) # type: ignore

@register_atom(witness_design_resampling_filter, name="design_resampling_filter")
@icontract.require(lambda up, down, numtaps: up > 0, "Precondition failed: up > 0")
@icontract.require(lambda up, down, numtaps: down > 0, "Precondition failed: down > 0")
@icontract.ensure(lambda result, up, down, numtaps: result.ndim == 1, "Postcondition failed: result.ndim == 1")
def design_resampling_filter(up: int, down: int, numtaps: int = None) -> NDArray[np.float64]:
    """Designs an interpolation filter for rational resampling.

    Args:
        up: int
        down: int
        numtaps: int

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.firwin(up=up, down=down, numtaps=numtaps) # type: ignore

@register_atom(witness_apply_polyphase_filtering, name="apply_polyphase_filtering")
@icontract.require(lambda signal, up, down, window_coeffs: signal.ndim == 1, "Precondition failed: signal.ndim == 1")
@icontract.require(lambda signal, up, down, window_coeffs: window_coeffs.ndim == 1, "Precondition failed: window_coeffs.ndim == 1")
@icontract.ensure(lambda result, signal, up, down, window_coeffs: result.ndim == signal.ndim, "Postcondition failed: result.ndim == signal.ndim")
def apply_polyphase_filtering(signal: NDArray[np.float64], up: int, down: int, window_coeffs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies rational resampling using polyphase decomposition.

    Args:
        signal: NDArray[np.float64]
        up: int
        down: int
        window_coeffs: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.resample_poly(signal=signal, up=up, down=down, window_coeffs=window_coeffs) # type: ignore

