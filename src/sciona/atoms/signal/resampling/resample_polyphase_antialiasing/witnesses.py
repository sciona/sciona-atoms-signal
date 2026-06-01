from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_rational_approximation(up: AbstractScalar | float, down: AbstractScalar | float, max_len: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for compute_rational_approximation."""
    _ = (up, down, max_len)
    return AbstractScalar(dtype="float64")

def witness_design_resampling_filter(up: AbstractScalar | int, down: AbstractScalar | int, numtaps: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for design_resampling_filter."""
    _ = (up, down, numtaps)
    return AbstractArray(shape=(), dtype="float64")

def witness_apply_polyphase_filtering(signal: AbstractArray, up: AbstractScalar | int, down: AbstractScalar | int, window_coeffs: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_polyphase_filtering."""
    _ = (signal, up, down, window_coeffs)
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)

