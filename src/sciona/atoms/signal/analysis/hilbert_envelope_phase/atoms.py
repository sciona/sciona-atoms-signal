from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_analytic_signal,
)

@register_atom(witness_compute_analytic_signal, name="compute_analytic_signal")
@icontract.require(lambda signal: signal.ndim == 1, "Precondition failed: signal.ndim == 1")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "Postcondition failed: result.shape == signal.shape")
def compute_analytic_signal(signal: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Computes the complex analytic signal of a real sequence via the Hilbert transform.

    Args:
        signal: NDArray[np.float64]

    Returns:
        result: NDArray[np.complex128]
    """
    import scipy.signal
    return scipy.signal.hilbert(signal=signal) # type: ignore

