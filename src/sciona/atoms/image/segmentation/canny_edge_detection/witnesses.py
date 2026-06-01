from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_image_gradients(image: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_image_gradients."""
    _ = (image)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

def witness_apply_non_maximum_suppression(grad_x: AbstractArray, grad_y: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_non_maximum_suppression."""
    _ = (grad_x, grad_y)
    return AbstractArray(shape=grad_x.shape, dtype=grad_x.dtype)

def witness_apply_hysteresis_thresholding(suppressed_image: AbstractArray, low_threshold: AbstractScalar | float, high_threshold: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for apply_hysteresis_thresholding."""
    _ = (suppressed_image, low_threshold, high_threshold)
    return AbstractArray(shape=suppressed_image.shape, dtype=suppressed_image.dtype)

