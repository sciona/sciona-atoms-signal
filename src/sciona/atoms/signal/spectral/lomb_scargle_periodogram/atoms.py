from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_lomb_scargle_power,
)

@register_atom(witness_compute_lomb_scargle_power, name="compute_lomb_scargle_power")
@icontract.require(lambda t, y, freqs: t.ndim == 1, "Precondition failed: t.ndim == 1")
@icontract.require(lambda t, y, freqs: y.ndim == 1, "Precondition failed: y.ndim == 1")
@icontract.require(lambda t, y, freqs: t.shape[0] == y.shape[0], "Precondition failed: t.shape[0] == y.shape[0]")
@icontract.require(lambda t, y, freqs: np.all(np.diff(t) > 0), "Precondition failed: np.all(np.diff(t) > 0)")
@icontract.ensure(lambda result, t, y, freqs: result.shape == freqs.shape, "Postcondition failed: result.shape == freqs.shape")
@icontract.ensure(lambda result, t, y, freqs: np.all(result >= 0), "Postcondition failed: np.all(result >= 0)")
def compute_lomb_scargle_power(t: NDArray[np.float64], y: NDArray[np.float64], freqs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the Lomb-Scargle power spectrum for irregularly spaced time series.

    Args:
        t: 1D array of sample times
        y: 1D array of observations
        freqs: 1D array of target frequencies

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.lombscargle(t=t, y=y, freqs=freqs) # type: ignore

