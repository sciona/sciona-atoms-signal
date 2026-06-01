from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_tile_cdfs,
)

@register_atom(witness_compute_tile_cdfs, name="compute_tile_cdfs")
@icontract.require(lambda image, kernel_size, clip_limit: image.ndim == 2, "Precondition failed: image.ndim == 2")
@icontract.require(lambda image, kernel_size, clip_limit: clip_limit >= 0, "Precondition failed: clip_limit >= 0")
@icontract.ensure(lambda result, image, kernel_size, clip_limit: result.ndim == 3, "Postcondition failed: result.ndim == 3")
def compute_tile_cdfs(image: NDArray[np.float64], kernel_size: int, clip_limit: float) -> NDArray[np.float64]:
    """Computes local, contrast-limited cumulative distribution functions in a grid.

    Args:
        image: NDArray[np.float64]
        kernel_size: tuple[int, int]
        clip_limit: float

    Returns:
        result: NDArray[np.float64]
    """
    import skimage.exposure
    return skimage.exposure.equalize_adapthist(image=image, kernel_size=kernel_size, clip_limit=clip_limit) # type: ignore

