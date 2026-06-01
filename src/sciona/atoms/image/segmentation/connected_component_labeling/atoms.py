from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_label_binary_components,
)

@register_atom(witness_label_binary_components, name="label_binary_components")
@icontract.require(lambda binary_image, connectivity: binary_image.ndim == 2, "Precondition failed: binary_image.ndim == 2")
@icontract.ensure(lambda result, binary_image, connectivity: labeled_image.shape == binary_image.shape, "Postcondition failed: labeled_image.shape == binary_image.shape")
@icontract.ensure(lambda result, binary_image, connectivity: num_features >= 0, "Postcondition failed: num_features >= 0")
def label_binary_components(binary_image: NDArray[np.bool_], connectivity: int = None) -> NDArray[np.int64]:
    """Labels connected components in a binary image.

    Args:
        binary_image: NDArray[np.bool_]
        connectivity: int

    Returns:
        labeled_image: NDArray[np.int64]
    """
    import scipy.ndimage
    return scipy.ndimage.label(binary_image=binary_image, connectivity=connectivity) # type: ignore

