from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_log_scale_space,
)

@register_atom(witness_generate_log_scale_space, name="generate_log_scale_space")
@icontract.require(lambda image, sigmas: image.ndim == 2, "Precondition failed: image.ndim == 2")
@icontract.require(lambda image, sigmas: sigmas.ndim == 1, "Precondition failed: sigmas.ndim == 1")
@icontract.ensure(lambda result, image, sigmas: result.ndim == 3, "Postcondition failed: result.ndim == 3")
@icontract.ensure(lambda result, image, sigmas: result.shape[0] == sigmas.shape[0], "Postcondition failed: result.shape[0] == sigmas.shape[0]")
def generate_log_scale_space(image: NDArray[np.float64], sigmas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generates a 3D volume of scale-normalized Laplacian of Gaussian filters.

    Args:
        image: NDArray[np.float64]
        sigmas: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import skimage.feature
    return skimage.feature.blob_log(image=image, sigmas=sigmas) # type: ignore

