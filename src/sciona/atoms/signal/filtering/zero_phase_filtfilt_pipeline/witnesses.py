from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_gustafsson_initial_conditions(b: AbstractArray, a: AbstractArray, signal: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_gustafsson_initial_conditions."""
    _ = (b, a, signal)
    return AbstractArray(shape=b.shape, dtype=b.dtype)

def witness_apply_zero_phase_forward_backward(b: AbstractArray, a: AbstractArray, signal: AbstractArray, zi: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_zero_phase_forward_backward."""
    _ = (b, a, signal, zi)
    return AbstractArray(shape=b.shape, dtype=b.dtype)

