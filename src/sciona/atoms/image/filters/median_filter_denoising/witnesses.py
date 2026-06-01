from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_apply_median_filtering_pass(image: AbstractArray, size: AbstractScalar | int, mode: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for apply_median_filtering_pass."""
    _ = (image, size, mode)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

