from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_convolving_padding,
    witness_multiply_spectral_coefficients,
)

@register_atom(witness_compute_convolving_padding, name="compute_convolving_padding")
@icontract.require(lambda in1_shape, in2_shape: len(in1_shape) == len(in2_shape), "Precondition failed: len(in1_shape) == len(in2_shape)")
@icontract.ensure(lambda result, in1_shape, in2_shape: all(r >= in1 + in2 - 1 for r, in1, in2 in zip(result, in1_shape, in2_shape)), "Postcondition failed: all(r >= in1 + in2 - 1 for r, in1, in2 in zip(result, in1_shape, in2_shape))")
def compute_convolving_padding(in1_shape: int, in2_shape: int) -> int:
    """Determines the optimal zero-padding size for linear convolution of two arrays.

    Args:
        in1_shape: tuple[int, ...]
        in2_shape: tuple[int, ...]

    Returns:
        result: tuple[int, ...]
    """
    import scipy.signal
    return scipy.signal.fftconvolve(in1_shape=in1_shape, in2_shape=in2_shape) # type: ignore

@register_atom(witness_multiply_spectral_coefficients, name="multiply_spectral_coefficients")
@icontract.require(lambda spec1, spec2: spec1.shape == spec2.shape, "Precondition failed: spec1.shape == spec2.shape")
@icontract.ensure(lambda result, spec1, spec2: result.shape == spec1.shape, "Postcondition failed: result.shape == spec1.shape")
def multiply_spectral_coefficients(spec1: NDArray[np.complex128], spec2: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Element-wise multiplication of two complex spectra in the frequency domain.

    Args:
        spec1: NDArray[np.complex128]
        spec2: NDArray[np.complex128]

    Returns:
        result: NDArray[np.complex128]
    """
    import numpy
    return numpy.multiply(spec1=spec1, spec2=spec2) # type: ignore

