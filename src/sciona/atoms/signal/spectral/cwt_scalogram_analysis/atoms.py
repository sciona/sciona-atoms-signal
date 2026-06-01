from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_cwt_kernels,
)

@register_atom(witness_generate_cwt_kernels, name="generate_cwt_kernels")
@icontract.require(lambda scales, wavelet_type: scales.ndim == 1, "Precondition failed: scales.ndim == 1")
@icontract.require(lambda scales, wavelet_type: np.all(scales > 0), "Precondition failed: np.all(scales > 0)")
@icontract.ensure(lambda result, scales, wavelet_type: len(result) == scales.shape[0], "Postcondition failed: len(result) == scales.shape[0]")
def generate_cwt_kernels(scales: NDArray[np.float64], wavelet_type: str) -> list[NDArray[np.float64]]:
    """Generates normalized 1D wavelet kernels for a set of continuous scales.

    Args:
        scales: NDArray[np.float64]
        wavelet_type: str

    Returns:
        result: list[NDArray[np.float64]]
    """
    import scipy.signal
    return scipy.signal.morlet2(scales=scales, wavelet_type=wavelet_type) # type: ignore

