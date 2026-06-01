from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_hough_accumulator(binary_image: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_hough_accumulator."""
    _ = (binary_image)
    return AbstractArray(shape=binary_image.shape, dtype=binary_image.dtype)

