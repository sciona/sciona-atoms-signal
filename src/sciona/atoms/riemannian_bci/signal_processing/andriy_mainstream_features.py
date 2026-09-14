# SPDX-License-Identifier: GPL-2.0-or-later
"""Assemble the Andriy 102-feature vector with documented Welch defaults."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .andriy_time_features import andriy_time_features
from .andriy_spectral_features import andriy_spectral_features
from .andriy_distribution_features import andriy_distribution_features
from .andriy_nonlinear_features import andriy_nonlinear_features
from .andriy_wavelet_features import andriy_wavelet_features
from .andriy_welch_features import andriy_documented_welch_features


def witness_andriy_documented_mainstream_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 102), dtype='float64')


@register_atom(witness_andriy_documented_mainstream_features)
def andriy_documented_mainstream_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return 102 columns in SPC_extract_features order at 256 Hz.

    Each input row is a finite window with length a positive multiple of 256.
    Range strictly below .01 yields an all-NaN row. The source computes those
    rows on random replacement signals before masking; this pure implementation
    directly masks without consuming RNG. It does not reproduce global RNG side
    effects. The caller's range<1000 gate, preprocessing, clip feature imputation
    and model execution are separate. Welch uses current documented MATLAB
    defaults; original historical MATLAB numerical parity remains unproven.
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 256 or x.shape[1] % 256 or not np.all(np.isfinite(x)):
        raise ValueError('finite windows/samples with length a positive multiple of 256 required')
    x = x.astype(float)
    output = np.full((len(x), 102), np.nan)
    valid = np.ptp(x, axis=1) >= .01
    if not np.any(valid):
        return output
    selected = x[valid]
    time = andriy_time_features(selected)
    spectral = andriy_spectral_features(selected)
    distribution = andriy_distribution_features(selected)
    nonlinear = andriy_nonlinear_features(selected)
    wavelet = andriy_wavelet_features(selected)
    welch = andriy_documented_welch_features(selected)
    leading = np.column_stack([
        nonlinear[:, 1], nonlinear[:, 2], time[:, 1], time[:, 2], spectral[:, 80],
        time[:, 3:6], nonlinear[:, 0], spectral[:, 81], time[:, 0], wavelet,
        distribution, time[:, 6:10]])
    output[valid] = np.column_stack([leading, spectral[:, :66], welch, spectral[:, 66:80]])
    return output
