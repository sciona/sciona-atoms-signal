"""Wavelet-based signal denoising via discrete wavelet transform."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_wavelet_denoise


@register_atom(witness_wavelet_denoise)
@icontract.require(lambda signal: signal.ndim == 1, "Input must be 1D signal")
@icontract.require(lambda level: level >= 1, "Decomposition level must be >= 1")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "Output shape preserved")
def wavelet_denoise(
    signal: NDArray[np.float64],
    wavelet: str = "db4",
    level: int = 4,
    threshold_mode: str = "soft",
) -> NDArray[np.float64]:
    """Denoise a 1D signal using discrete wavelet transform with universal thresholding.

    Decomposes signal via DWT, applies universal threshold (sigma * sqrt(2 * log(n)))
    to detail coefficients, then reconstructs.
    """
    import pywt

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(len(signal)))

    # Threshold detail coefficients (keep approximation unchanged)
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        if threshold_mode == "soft":
            denoised = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        else:
            denoised = detail * (np.abs(detail) >= threshold)
        denoised_coeffs.append(denoised)

    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)]
