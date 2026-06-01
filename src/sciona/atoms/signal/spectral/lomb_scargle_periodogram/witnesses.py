from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_lomb_scargle_power(t: AbstractArray, y: AbstractArray, freqs: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_lomb_scargle_power."""
    _ = (t, y, freqs)
    return AbstractArray(shape=t.shape, dtype=t.dtype)

