from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_convolving_padding(in1_shape: AbstractScalar | int, in2_shape: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for compute_convolving_padding."""
    _ = (in1_shape, in2_shape)
    return AbstractScalar(dtype="float64")

def witness_multiply_spectral_coefficients(spec1: AbstractArray, spec2: AbstractArray) -> AbstractArray:
    """Ghost witness for multiply_spectral_coefficients."""
    _ = (spec1, spec2)
    return AbstractArray(shape=spec1.shape, dtype=spec1.dtype)

