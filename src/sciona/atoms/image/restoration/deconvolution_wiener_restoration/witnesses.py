from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_psf_to_otf(psf: AbstractArray, target_shape: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for compute_psf_to_otf."""
    _ = (psf, target_shape)
    return AbstractArray(shape=psf.shape, dtype=psf.dtype)

def witness_compute_wiener_filter_kernel(otf: AbstractArray, nsr: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for compute_wiener_filter_kernel."""
    _ = (otf, nsr)
    return AbstractArray(shape=otf.shape, dtype=otf.dtype)

def witness_apply_spectral_wiener_filtering(image: AbstractArray, filter_kernel: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_spectral_wiener_filtering."""
    _ = (image, filter_kernel)
    return AbstractArray(shape=image.shape, dtype=image.dtype)

