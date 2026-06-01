from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_hierarchical_dwt,
)

@register_atom(witness_compute_hierarchical_dwt, name="compute_hierarchical_dwt")
@icontract.require(lambda data, wavelet, level: data.ndim == 1, "Precondition failed: data.ndim == 1")
@icontract.ensure(lambda result, data, wavelet, level: len(result) > 1, "Postcondition failed: len(result) > 1")
def compute_hierarchical_dwt(data: NDArray[np.float64], wavelet: str, level: int = None) -> list[NDArray[np.float64]]:
    """Computes 1D discrete wavelet decomposition coefficients down to a specified level.

    Args:
        data: NDArray[np.float64]
        wavelet: str
        level: int

    Returns:
        result: list[NDArray[np.float64]]
    """
    import pywt
    return pywt.wavedec(data=data, wavelet=wavelet, level=level) # type: ignore

