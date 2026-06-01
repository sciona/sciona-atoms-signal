from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_analytic_signal(signal: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_analytic_signal."""
    _ = (signal)
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)

