from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_hierarchical_dwt(data: AbstractArray, wavelet: AbstractScalar | str, level: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for compute_hierarchical_dwt."""
    _ = (data, wavelet, level)
    return AbstractArray(shape=data.shape, dtype=data.dtype)

