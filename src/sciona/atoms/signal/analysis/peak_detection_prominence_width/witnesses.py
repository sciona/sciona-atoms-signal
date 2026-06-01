from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_find_local_maxima(signal: AbstractArray) -> AbstractArray:
    """Ghost witness for find_local_maxima."""
    _ = (signal)
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)

def witness_compute_peak_prominences(signal: AbstractArray, peaks: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_peak_prominences."""
    _ = (signal, peaks)
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)

def witness_compute_peak_widths(signal: AbstractArray, peaks: AbstractArray, prominences: AbstractArray, rel_height: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for compute_peak_widths."""
    _ = (signal, peaks, prominences, rel_height)
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)

