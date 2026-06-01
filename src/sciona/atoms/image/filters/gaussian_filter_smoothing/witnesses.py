from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_gaussian_1d_kernel(sigma: AbstractScalar | float, truncate: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for compute_gaussian_1d_kernel."""
    _ = (sigma, truncate)
    return AbstractArray(shape=(), dtype="float64")

def witness_apply_separable_convolution_2d(image: AbstractArray, kernel: AbstractArray, mode: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for apply_separable_convolution_2d."""
    _ = (image, kernel, mode)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

