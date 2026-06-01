from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_apply_median_filtering_pass,
)

@register_atom(witness_apply_median_filtering_pass, name="apply_median_filtering_pass")
@icontract.require(lambda image, size, mode: image.ndim >= 2, "Precondition failed: image.ndim >= 2")
@icontract.require(lambda image, size, mode: size > 0, "Precondition failed: size > 0")
@icontract.require(lambda image, size, mode: size % 2 == 1, "Precondition failed: size % 2 == 1")
@icontract.ensure(lambda result, image, size, mode: result.shape == image.shape, "Postcondition failed: result.shape == image.shape")
def apply_median_filtering_pass(image: NDArray[np.float64], size: int, mode: str = None) -> NDArray[np.float64]:
    """Applies a sliding-window median filter to a multi-dimensional array.

    Args:
        image: NDArray[np.float64]
        size: int
        mode: str

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.ndimage
    return scipy.ndimage.median_filter(image=image, size=size, mode=mode) # type: ignore

