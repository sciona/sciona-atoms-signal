from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_design_iir_sos_coefficients(order: AbstractScalar | int, Wn: AbstractArray, btype: AbstractScalar | str, ftype: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for design_iir_sos_coefficients."""
    _ = (order, Wn, btype, ftype)
    return AbstractArray(shape=Wn.shape, dtype=Wn.dtype)

def witness_apply_iir_sosfilter(sos: AbstractArray, signal: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_iir_sosfilter."""
    _ = (sos, signal)
    return AbstractArray(shape=sos.shape, dtype=sos.dtype)

