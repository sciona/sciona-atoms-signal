from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_savgol_coefficients(window_length: AbstractScalar | int, polyorder: AbstractScalar | int, deriv: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for compute_savgol_coefficients."""
    _ = (window_length, polyorder, deriv)
    return AbstractArray(shape=(), dtype="float64")

def witness_apply_savgol_convolution(signal: AbstractArray, coeffs: AbstractArray, mode: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for apply_savgol_convolution."""
    _ = (signal, coeffs, mode)
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)

