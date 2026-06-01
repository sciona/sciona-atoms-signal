from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_distance_transform(binary_mask: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_distance_transform."""
    _ = (binary_mask)
    return AbstractArray(shape=binary_mask.shape, dtype=binary_mask.dtype)

def witness_find_watershed_markers(distance_map: AbstractArray, min_distance: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for find_watershed_markers."""
    _ = (distance_map, min_distance)
    return AbstractArray(shape=distance_map.shape, dtype=distance_map.dtype)

def witness_apply_watershed_flooding(image: AbstractArray, markers: AbstractArray, mask: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_watershed_flooding."""
    _ = (image, markers, mask)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

