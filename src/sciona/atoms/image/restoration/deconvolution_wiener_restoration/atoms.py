from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_psf_to_otf,
    witness_compute_wiener_filter_kernel,
    witness_apply_spectral_wiener_filtering,
)

@register_atom(witness_compute_psf_to_otf, name="compute_psf_to_otf")
@icontract.require(lambda psf, target_shape: psf.ndim == 2, "Precondition failed: psf.ndim == 2")
@icontract.ensure(lambda result, psf, target_shape: result.shape == target_shape, "Postcondition failed: result.shape == target_shape")
def compute_psf_to_otf(psf: NDArray[np.float64], target_shape: int) -> NDArray[np.complex128]:
    """Converts a spatial Point Spread Function (PSF) to the frequency-domain Optical Transfer Function (OTF).

    Args:
        psf: 2D spatial kernel
        target_shape: tuple[int, int]

    Returns:
        result: NDArray[np.complex128]
    """
    import scipy.fft
    return scipy.fft.fft2(psf=psf, target_shape=target_shape) # type: ignore

@register_atom(witness_compute_wiener_filter_kernel, name="compute_wiener_filter_kernel")
@icontract.require(lambda otf, nsr: nsr >= 0, "Precondition failed: nsr >= 0")
@icontract.ensure(lambda result, otf, nsr: result.shape == otf.shape, "Postcondition failed: result.shape == otf.shape")
def compute_wiener_filter_kernel(otf: NDArray[np.complex128], nsr: float) -> NDArray[np.complex128]:
    """Computes the Wiener deconvolution transfer function.

    Args:
        otf: NDArray[np.complex128]
        nsr: Noise-to-signal ratio, positive float

    Returns:
        result: NDArray[np.complex128]
    """
    import skimage.restoration
    return skimage.restoration.wiener(otf=otf, nsr=nsr) # type: ignore

@register_atom(witness_apply_spectral_wiener_filtering, name="apply_spectral_wiener_filtering")
@icontract.require(lambda image, filter_kernel: image.ndim == 2, "Precondition failed: image.ndim == 2")
@icontract.require(lambda image, filter_kernel: filter_kernel.shape == image.shape, "Precondition failed: filter_kernel.shape == image.shape")
@icontract.ensure(lambda result, image, filter_kernel: result.shape == image.shape, "Postcondition failed: result.shape == image.shape")
def apply_spectral_wiener_filtering(image: NDArray[np.float64], filter_kernel: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Applies the Wiener filter to the blurred image in the frequency domain.

    Args:
        image: NDArray[np.float64]
        filter_kernel: NDArray[np.complex128]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.fft
    return scipy.fft.fft2(image=image, filter_kernel=filter_kernel) # type: ignore

