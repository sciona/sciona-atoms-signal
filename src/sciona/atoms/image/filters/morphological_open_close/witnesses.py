from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_structuring_element(shape: AbstractScalar | str, radius: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for generate_structuring_element."""
    _ = (shape, radius)
    return AbstractArray(shape=(), dtype="float64")

