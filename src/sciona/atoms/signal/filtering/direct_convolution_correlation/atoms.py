from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_pad_array_boundaries,
)

@register_atom(witness_pad_array_boundaries, name="pad_array_boundaries")
@icontract.require(lambda array, pad_width, mode: array.ndim == len(pad_width), "Precondition failed: array.ndim == len(pad_width)")
@icontract.ensure(lambda result, array, pad_width, mode: result.ndim == array.ndim, "Postcondition failed: result.ndim == array.ndim")
def pad_array_boundaries(array: NDArray[np.float64], pad_width: int, mode: str) -> NDArray[np.float64]:
    """Applies chosen boundary conditions (e.g. constant, wrap, symmetric) to pad an array prior to spatial filtering.

    Args:
        array: NDArray[np.float64]
        pad_width: tuple[tuple[int, int], ...]
        mode: str

    Returns:
        result: NDArray[np.float64]
    """
    import numpy
    return numpy.pad(array=array, pad_width=pad_width, mode=mode) # type: ignore

