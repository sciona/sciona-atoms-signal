from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_design_fir_coefficients_window,
    witness_apply_fir_lfilter,
)

@register_atom(witness_design_fir_coefficients_window, name="design_fir_coefficients_window")
@icontract.require(lambda numtaps, cutoff, window_type: numtaps > 0, "Precondition failed: numtaps > 0")
@icontract.ensure(lambda result, numtaps, cutoff, window_type: result.shape[0] == numtaps, "Postcondition failed: result.shape[0] == numtaps")
def design_fir_coefficients_window(numtaps: int, cutoff: float | NDArray[np.float64], window_type: str = None) -> NDArray[np.float64]:
    """Designs linear-phase FIR filter coefficients using the window method.

    Args:
        numtaps: Positive integer, odd for linear phase type I
        cutoff: Normalized cutoff frequency (0 to 1)
        window_type: str

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, window_type=window_type) # type: ignore

@register_atom(witness_apply_fir_lfilter, name="apply_fir_lfilter")
@icontract.require(lambda b, signal: b.ndim == 1, "Precondition failed: b.ndim == 1")
@icontract.require(lambda b, signal: signal.ndim >= 1, "Precondition failed: signal.ndim >= 1")
@icontract.ensure(lambda result, b, signal: result.shape == signal.shape, "Postcondition failed: result.shape == signal.shape")
def apply_fir_lfilter(b: NDArray[np.float64], signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies a designed FIR filter to a signal using direct-form II transposed structure.

    Args:
        b: Filter numerator coefficients (FIR taps)
        signal: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.lfilter(b=b, signal=signal) # type: ignore

