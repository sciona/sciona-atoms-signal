from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_cwt_kernels(scales: AbstractArray, wavelet_type: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for generate_cwt_kernels."""
    _ = (scales, wavelet_type)
    return AbstractArray(shape=scales.shape, dtype=scales.dtype)

