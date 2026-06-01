from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_correlation_map(image: AbstractArray, template: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_correlation_map."""
    _ = (image, template)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

