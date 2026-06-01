from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_tile_cdfs(image: AbstractArray, kernel_size: AbstractScalar | int, clip_limit: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for compute_tile_cdfs."""
    _ = (image, kernel_size, clip_limit)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

