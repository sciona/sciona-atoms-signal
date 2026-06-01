from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_gaussian_1d_kernel,
    witness_apply_separable_convolution_2d,
)

@register_atom(witness_compute_gaussian_1d_kernel, name="compute_gaussian_1d_kernel")
@icontract.require(lambda sigma, truncate: sigma > 0, "Precondition failed: sigma > 0")
@icontract.require(lambda sigma, truncate: truncate > 0, "Precondition failed: truncate > 0")
@icontract.ensure(lambda result, sigma, truncate: result.ndim == 1, "Postcondition failed: result.ndim == 1")
@icontract.ensure(lambda result, sigma, truncate: result.shape[0] % 2 == 1, "Postcondition failed: result.shape[0] % 2 == 1")
def compute_gaussian_1d_kernel(sigma: float, truncate: float = None) -> NDArray[np.float64]:
    """Computes a normalized 1D Gaussian kernel for a given standard deviation.

    Args:
        sigma: Positive float
        truncate: Truncation radius in standard deviations (default 4.0)

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.ndimage
    return scipy.ndimage.gaussian_filter(sigma=sigma, truncate=truncate) # type: ignore

@register_atom(witness_apply_separable_convolution_2d, name="apply_separable_convolution_2d")
@icontract.require(lambda image, kernel, mode: image.ndim == 2, "Precondition failed: image.ndim == 2")
@icontract.require(lambda image, kernel, mode: kernel.ndim == 1, "Precondition failed: kernel.ndim == 1")
@icontract.ensure(lambda result, image, kernel, mode: result.shape == image.shape, "Postcondition failed: result.shape == image.shape")
def apply_separable_convolution_2d(image: NDArray[np.float64], kernel: NDArray[np.float64], mode: str = None) -> NDArray[np.float64]:
    """Convolves a 2D image sequentially along rows and columns with 1D kernels.

    Args:
        image: 2D image
        kernel: NDArray[np.float64]
        mode: str

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.ndimage
    return scipy.ndimage.convolve1d(image=image, kernel=kernel, mode=mode) # type: ignore

