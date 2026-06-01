from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_hough_accumulator,
)

@register_atom(witness_compute_hough_accumulator, name="compute_hough_accumulator")
@icontract.require(lambda binary_image: binary_image.ndim == 2, "Precondition failed: binary_image.ndim == 2")
@icontract.ensure(lambda result, binary_image: accumulator.ndim == 2, "Postcondition failed: accumulator.ndim == 2")
def compute_hough_accumulator(binary_image: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Populates the 2D Hough parameter accumulator from a binary edge image.

    Args:
        binary_image: NDArray[np.bool_]

    Returns:
        accumulator: NDArray[np.int64]
    """
    import skimage.transform
    return skimage.transform.hough_line(binary_image=binary_image) # type: ignore

