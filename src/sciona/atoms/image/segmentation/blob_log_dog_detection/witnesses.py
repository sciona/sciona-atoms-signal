from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_log_scale_space(image: AbstractArray, sigmas: AbstractArray) -> AbstractArray:
    """Ghost witness for generate_log_scale_space."""
    _ = (image, sigmas)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

