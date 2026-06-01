from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_gustafsson_initial_conditions,
    witness_apply_zero_phase_forward_backward,
)

@register_atom(witness_compute_gustafsson_initial_conditions, name="compute_gustafsson_initial_conditions")
@icontract.require(lambda b, a, signal: b.ndim == 1, "Precondition failed: b.ndim == 1")
@icontract.require(lambda b, a, signal: a.ndim == 1, "Precondition failed: a.ndim == 1")
@icontract.require(lambda b, a, signal: signal.ndim >= 1, "Precondition failed: signal.ndim >= 1")
@icontract.ensure(lambda result, b, a, signal: result.ndim == 1, "Postcondition failed: result.ndim == 1")
def compute_gustafsson_initial_conditions(b: NDArray[np.float64], a: NDArray[np.float64], signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes initial conditions for forward-backward filtering to minimize edge transients.

    Args:
        b: NDArray[np.float64]
        a: NDArray[np.float64]
        signal: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.lfiltic(b=b, a=a, signal=signal) # type: ignore

@register_atom(witness_apply_zero_phase_forward_backward, name="apply_zero_phase_forward_backward")
@icontract.require(lambda b, a, signal, zi: b.ndim == 1, "Precondition failed: b.ndim == 1")
@icontract.require(lambda b, a, signal, zi: a.ndim == 1, "Precondition failed: a.ndim == 1")
@icontract.ensure(lambda result, b, a, signal, zi: result.shape == signal.shape, "Postcondition failed: result.shape == signal.shape")
def apply_zero_phase_forward_backward(b: NDArray[np.float64], a: NDArray[np.float64], signal: NDArray[np.float64], zi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies double-filtering forward and backward with padding.

    Args:
        b: NDArray[np.float64]
        a: NDArray[np.float64]
        signal: NDArray[np.float64]
        zi: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.filtfilt(b=b, a=a, signal=signal, zi=zi) # type: ignore

