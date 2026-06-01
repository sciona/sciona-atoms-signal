from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_savgol_coefficients,
    witness_apply_savgol_convolution,
)

@register_atom(witness_compute_savgol_coefficients, name="compute_savgol_coefficients")
@icontract.require(lambda window_length, polyorder, deriv: window_length % 2 == 1, "Precondition failed: window_length % 2 == 1")
@icontract.require(lambda window_length, polyorder, deriv: window_length > polyorder, "Precondition failed: window_length > polyorder")
@icontract.require(lambda window_length, polyorder, deriv: deriv >= 0, "Precondition failed: deriv >= 0")
@icontract.ensure(lambda result, window_length, polyorder, deriv: result.shape[0] == window_length, "Postcondition failed: result.shape[0] == window_length")
def compute_savgol_coefficients(window_length: int, polyorder: int, deriv: int = None) -> NDArray[np.float64]:
    """Computes the convolution coefficients for Savitzky-Golay smoothing and derivatives.

    Args:
        window_length: Must be a positive odd integer
        polyorder: Must be less than window_length
        deriv: Order of derivative (default 0)

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.savgol_coeffs(window_length=window_length, polyorder=polyorder, deriv=deriv) # type: ignore

@register_atom(witness_apply_savgol_convolution, name="apply_savgol_convolution")
@icontract.require(lambda signal, coeffs, mode: signal.ndim == 1, "Precondition failed: signal.ndim == 1")
@icontract.require(lambda signal, coeffs, mode: coeffs.ndim == 1, "Precondition failed: coeffs.ndim == 1")
@icontract.ensure(lambda result, signal, coeffs, mode: result.shape == signal.shape, "Postcondition failed: result.shape == signal.shape")
def apply_savgol_convolution(signal: NDArray[np.float64], coeffs: NDArray[np.float64], mode: str = None) -> NDArray[np.float64]:
    """Convolves a signal with Savitzky-Golay coefficients, handling boundary effects.

    Args:
        signal: NDArray[np.float64]
        coeffs: NDArray[np.float64]
        mode: str

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.savgol_filter(signal=signal, coeffs=coeffs, mode=mode) # type: ignore

