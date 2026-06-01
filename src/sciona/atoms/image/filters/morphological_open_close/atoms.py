from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_structuring_element,
)

@register_atom(witness_generate_structuring_element, name="generate_structuring_element")
@icontract.require(lambda shape, radius: radius > 0, "Precondition failed: radius > 0")
@icontract.ensure(lambda result, shape, radius: result.ndim == 2, "Postcondition failed: result.ndim == 2")
@icontract.ensure(lambda result, shape, radius: result.shape[0] % 2 == 1, "Postcondition failed: result.shape[0] % 2 == 1")
def generate_structuring_element(shape: str, radius: int) -> NDArray[np.bool_]:
    """Creates a binary footprint mask of a chosen shape and size.

    Args:
        shape: str
        radius: int

    Returns:
        result: NDArray[np.bool_]
    """
    import skimage.morphology
    return skimage.morphology.disk(shape=shape, radius=radius) # type: ignore

