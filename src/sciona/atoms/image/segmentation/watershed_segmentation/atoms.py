from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_distance_transform,
    witness_find_watershed_markers,
    witness_apply_watershed_flooding,
)

@register_atom(witness_compute_distance_transform, name="compute_distance_transform")
@icontract.require(lambda binary_mask: binary_mask.ndim == 2, "Precondition failed: binary_mask.ndim == 2")
@icontract.ensure(lambda result, binary_mask: result.shape == binary_mask.shape, "Postcondition failed: result.shape == binary_mask.shape")
def compute_distance_transform(binary_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Computes the Euclidean distance transform of a binary mask to identify marker centers.

    Args:
        binary_mask: NDArray[np.bool_]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.ndimage
    return scipy.ndimage.distance_transform_edt(binary_mask=binary_mask) # type: ignore

@register_atom(witness_find_watershed_markers, name="find_watershed_markers")
@icontract.require(lambda distance_map, min_distance: distance_map.ndim == 2, "Precondition failed: distance_map.ndim == 2")
@icontract.ensure(lambda result, distance_map, min_distance: result.shape == distance_map.shape, "Postcondition failed: result.shape == distance_map.shape")
def find_watershed_markers(distance_map: NDArray[np.float64], min_distance: int = None) -> NDArray[np.int64]:
    """Identifies and labels local maxima of the distance transform as seeds.

    Args:
        distance_map: NDArray[np.float64]
        min_distance: int

    Returns:
        result: NDArray[np.int64]
    """
    import scipy.ndimage
    return scipy.ndimage.label(distance_map=distance_map, min_distance=min_distance) # type: ignore

@register_atom(witness_apply_watershed_flooding, name="apply_watershed_flooding")
@icontract.require(lambda image, markers, mask: image.ndim == 2, "Precondition failed: image.ndim == 2")
@icontract.require(lambda image, markers, mask: markers.shape == image.shape, "Precondition failed: markers.shape == image.shape")
@icontract.ensure(lambda result, image, markers, mask: result.shape == image.shape, "Postcondition failed: result.shape == image.shape")
def apply_watershed_flooding(image: NDArray[np.float64], markers: NDArray[np.int64], mask: NDArray[np.bool_] = None) -> NDArray[np.int64]:
    """Applies the watershed flooding algorithm from markers on a gradient surface.

    Args:
        image: Gradient magnitude or original image
        markers: NDArray[np.int64]
        mask: NDArray[np.bool_]

    Returns:
        result: NDArray[np.int64]
    """
    import skimage.segmentation
    return skimage.segmentation.watershed(image=image, markers=markers, mask=mask) # type: ignore

