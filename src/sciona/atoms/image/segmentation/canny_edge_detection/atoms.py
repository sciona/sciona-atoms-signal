from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_image_gradients,
    witness_apply_non_maximum_suppression,
    witness_apply_hysteresis_thresholding,
)

@register_atom(witness_compute_image_gradients, name="compute_image_gradients")
@icontract.require(lambda image: image.ndim == 2, "Precondition failed: image.ndim == 2")
@icontract.ensure(lambda result, image: grad_x.shape == image.shape, "Postcondition failed: grad_x.shape == image.shape")
@icontract.ensure(lambda result, image: grad_y.shape == image.shape, "Postcondition failed: grad_y.shape == image.shape")
def compute_image_gradients(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the horizontal and vertical image gradients using Sobel or Scharr operators.

    Args:
        image: NDArray[np.float64]

    Returns:
        grad_x: NDArray[np.float64]
    """
    import scipy.ndimage
    return scipy.ndimage.sobel(image=image) # type: ignore

@register_atom(witness_apply_non_maximum_suppression, name="apply_non_maximum_suppression")
@icontract.require(lambda grad_x, grad_y: grad_x.shape == grad_y.shape, "Precondition failed: grad_x.shape == grad_y.shape")
@icontract.ensure(lambda result, grad_x, grad_y: result.shape == grad_x.shape, "Postcondition failed: result.shape == grad_x.shape")
def apply_non_maximum_suppression(grad_x: NDArray[np.float64], grad_y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thins gradient magnitude maps along the direction of the gradient.

    Args:
        grad_x: NDArray[np.float64]
        grad_y: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import skimage.feature
    return skimage.feature.canny(grad_x=grad_x, grad_y=grad_y) # type: ignore

@register_atom(witness_apply_hysteresis_thresholding, name="apply_hysteresis_thresholding")
@icontract.require(lambda suppressed_image, low_threshold, high_threshold: low_threshold <= high_threshold, "Precondition failed: low_threshold <= high_threshold")
@icontract.ensure(lambda result, suppressed_image, low_threshold, high_threshold: result.shape == suppressed_image.shape, "Postcondition failed: result.shape == suppressed_image.shape")
def apply_hysteresis_thresholding(suppressed_image: NDArray[np.float64], low_threshold: float, high_threshold: float) -> NDArray[np.bool_]:
    """Applies low/high thresholds and tracks edges using connectivity.

    Args:
        suppressed_image: NDArray[np.float64]
        low_threshold: float
        high_threshold: float

    Returns:
        result: NDArray[np.bool_]
    """
    import skimage.filters
    return skimage.filters.apply_hysteresis_threshold(suppressed_image=suppressed_image, low_threshold=low_threshold, high_threshold=high_threshold) # type: ignore

