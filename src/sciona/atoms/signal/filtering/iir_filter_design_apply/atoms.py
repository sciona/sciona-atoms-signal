from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_design_iir_sos_coefficients,
    witness_apply_iir_sosfilter,
)

@register_atom(witness_design_iir_sos_coefficients, name="design_iir_sos_coefficients")
@icontract.require(lambda order, Wn, btype, ftype: order > 0, "Precondition failed: order > 0")
@icontract.ensure(lambda result, order, Wn, btype, ftype: result.ndim == 2, "Postcondition failed: result.ndim == 2")
@icontract.ensure(lambda result, order, Wn, btype, ftype: result.shape[1] == 6, "Postcondition failed: result.shape[1] == 6")
def design_iir_sos_coefficients(order: int, Wn: float | NDArray[np.float64], btype: str = None, ftype: str = None) -> NDArray[np.float64]:
    """Designs IIR filter coefficients in second-order sections (SOS) format for numerical stability.

    Args:
        order: int
        Wn: float | NDArray[np.float64]
        btype: str
        ftype: e.g. 'butter', 'cheby1', 'cheby2', 'ellip'

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.iirfilter(order=order, Wn=Wn, btype=btype, ftype=ftype) # type: ignore

@register_atom(witness_apply_iir_sosfilter, name="apply_iir_sosfilter")
@icontract.require(lambda sos, signal: sos.ndim == 2, "Precondition failed: sos.ndim == 2")
@icontract.require(lambda sos, signal: sos.shape[1] == 6, "Precondition failed: sos.shape[1] == 6")
@icontract.require(lambda sos, signal: signal.ndim >= 1, "Precondition failed: signal.ndim >= 1")
@icontract.ensure(lambda result, sos, signal: result.shape == signal.shape, "Postcondition failed: result.shape == signal.shape")
def apply_iir_sosfilter(sos: NDArray[np.float64], signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Filters a signal along one axis using second-order sections (SOS) format.

    Args:
        sos: L-by-6 matrix of second-order sections
        signal: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.sosfilt(sos=sos, signal=signal) # type: ignore

