# SPDX-License-Identifier: GPL-2.0-or-later
"""Andriy wavelet feature using the Uvi_Wave source transform conventions.

Algorithm provenance: Universidad de Vigo Uvi_Wave (1994–1996), GPL-2.0-or-later,
as distributed by the pinned public competition solution. This implementation
uses standard eight-tap Daubechies coefficients and preserves the source delay,
extension, decimation and subband selection conventions.
"""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


# Standard minimum-phase Daubechies four-vanishing-moment reconstruction filter.
_DB4 = np.array([.2303778133088964, .7148465705529154, .6308807679298587,
                 -.02798376941685985, -.18703481171909309, .030841381835560764,
                 .0328830116668852, -.010597401785069032])


def witness_andriy_wavelet_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 1), dtype='float64')


@register_atom(witness_andriy_wavelet_features)
def andriy_wavelet_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return source fifth-subband mean absolute coefficient, not squared energy.

    Seven scales use db4 analysis filters. Default source delays are one-based
    maximum-absolute-tap indices, with the highpass delay incremented when their
    difference is odd. Each scale appends zero for odd length before periodic
    extension and decimation. Source rounded subband boundaries depend on the
    original length, even when padding increases the transform length. Accept
    finite windows/samples matrices with at least 128 samples; preserve inputs.
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 128 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty windows/samples with at least 128 samples required')
    low = _DB4[::-1]
    high = _DB4 * (-1.)**np.arange(1, 9)
    low_delay = int(np.argmax(np.abs(low)))+1
    high_delay = int(np.argmax(np.abs(high)))+1
    high_delay += (high_delay-low_delay) % 2
    extension = max(len(low), len(high), low_delay, high_delay)
    result = []
    for row in x.astype(float):
        current = row
        details = []
        for _ in range(7):
            if len(current) % 2:
                current = np.append(current, 0.)
            length = len(current)
            wrapped = np.pad(current, (extension, extension), mode='wrap')
            detail = np.convolve(wrapped, high)[extension+high_delay:extension+high_delay+length:2]
            current = np.convolve(wrapped, low)[extension+low_delay:extension+low_delay+length:2]
            details.append(detail)
        coefficients = np.concatenate([current, *reversed(details)])
        start = int(np.floor(len(row)/32 + 1 + .5))-1
        stop = int(np.floor(len(row)/16 + 1 + .5))-1
        result.append([np.mean(np.abs(coefficients[start:stop]))])
    return np.asarray(result, dtype=float)
