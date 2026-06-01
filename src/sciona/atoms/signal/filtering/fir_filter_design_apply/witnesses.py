from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_design_fir_coefficients_window(numtaps: AbstractScalar | int, cutoff: AbstractArray, window_type: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for design_fir_coefficients_window."""
    _ = (numtaps, cutoff, window_type)
    return AbstractArray(shape=cutoff.shape, dtype=cutoff.dtype)

def witness_apply_fir_lfilter(b: AbstractArray, signal: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_fir_lfilter."""
    _ = (b, signal)
    return AbstractArray(shape=b.shape, dtype=b.dtype)

