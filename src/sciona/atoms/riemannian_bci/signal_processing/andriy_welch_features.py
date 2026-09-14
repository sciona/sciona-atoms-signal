"""Andriy Welch moments with explicitly documented MATLAB window defaults."""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_documented_welch_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 2), dtype='float64')


@register_atom(witness_andriy_documented_welch_features)
def andriy_documented_welch_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return mean frequency and bandwidth at 256 Hz under documented defaults.

    Explicit symmetric Hamming window length floor(N/4.5), half-window overlap
    rounded down, no detrending, and one-sided density. The source specifies
    next-power-of-two NFFT for the entire input and discards the final two bins
    before computing both moments. Zero power yields NaN. Finite nonempty
    windows/samples matrices, N>=32. This implements current documented MATLAB
    defaults, not native Octave defaults; historical MATLAB parity is unproven.
    Documentation: https://www.mathworks.com/help/signal/ref/pwelch.html
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 32 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty windows/samples with at least 32 samples required')
    samples = x.shape[1]
    length = int(np.floor(samples/4.5))
    nfft = 1 << (samples-1).bit_length()
    frequency, density = welch(x.astype(float), fs=256., window=np.hamming(length),
                               nperseg=length, noverlap=length//2, nfft=nfft,
                               detrend=False, return_onesided=True, scaling='density', axis=1)
    frequency, density = frequency[:-2], density[:, :-2]
    # Preserve the source's bin-index and spacing arithmetic.
    indices, spacing = frequency/256*nfft, 256/nfft
    total = np.sum(density, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = np.sum(density*indices*spacing, axis=1)/total
        bandwidth = np.sqrt(np.sum(density*(-indices*spacing+mean[:, None])**2, axis=1)/total)
    return np.column_stack([mean, bandwidth])
