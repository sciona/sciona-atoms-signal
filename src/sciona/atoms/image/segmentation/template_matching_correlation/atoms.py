from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_correlation_map,
)

@register_atom(witness_compute_correlation_map, name="compute_correlation_map")
@icontract.require(lambda image, template: image.ndim == 2, "Precondition failed: image.ndim == 2")
@icontract.require(lambda image, template: template.ndim == 2, "Precondition failed: template.ndim == 2")
@icontract.require(lambda image, template: template.shape[0] <= image.shape[0], "Precondition failed: template.shape[0] <= image.shape[0]")
@icontract.require(lambda image, template: template.shape[1] <= image.shape[1], "Precondition failed: template.shape[1] <= image.shape[1]")
@icontract.ensure(lambda result, image, template: np.all(result >= -1.0), "Postcondition failed: np.all(result >= -1.0)")
@icontract.ensure(lambda result, image, template: np.all(result <= 1.0), "Postcondition failed: np.all(result <= 1.0)")
def compute_correlation_map(image: NDArray[np.float64], template: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes 2D normalized cross-correlation coefficients map.

    Args:
        image: NDArray[np.float64]
        template: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import skimage.feature
    return skimage.feature.match_template(image=image, template=template) # type: ignore

