"""Source-compatible coherence between mean complex FFT band coefficients."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_frequency_band_coherence(windows: AbstractArray, frequency_bands: list,
                                     fs: float = 400., fft_window: int = 256,
                                     overlap: float = .5) -> AbstractArray:
    if len(windows.shape) != 3:
        raise ValueError('window/channel/sample tensor required')
    size = len(frequency_bands)
    return AbstractArray(shape=(windows.shape[0], size, size, windows.shape[1]), dtype='float64')


@register_atom(witness_frequency_band_coherence)
def frequency_band_coherence(windows: NDArray[np.float64], frequency_bands: list,
                             fs: float = 400., fft_window: int = 256,
                             overlap: float = .5) -> NDArray[np.float64]:
    """Return (windows, bands, bands, channels) normalized coherence matrices.

    Frame each channel with a symmetric Hann window and hop
    fft_window - int(overlap * fft_window). Drop incomplete trailing frames.
    Average complex FFT coefficients in each half-open [low, high) band BEFORE
    forming frame-averaged cross products and squared-magnitude normalization.
    This matches preproc.coherences(transpose=True, aggregate=False,
    normalize=True) in the pinned Barachant competition source. Channel count
    is generalized from its hardcoded 16. No dropped-sample filtering is applied.

    Bands must contain a positive-frequency-half FFT bin (Nyquist excluded).
    Inputs and band powers must be finite, and each channel/band must have
    strictly positive mean power. Matrices are positive semidefinite; strict
    positive definiteness is not guaranteed and no regularization is added.
    """
    a = np.asarray(windows)
    if a.dtype.kind not in 'iuf' or a.ndim != 3 or min(a.shape) == 0 or not np.all(np.isfinite(a)):
        raise ValueError('finite nonempty window/channel/sample tensor required')
    if isinstance(fft_window, bool) or not isinstance(fft_window, (int, np.integer)) or fft_window < 3:
        raise ValueError('integer FFT window of at least three samples required')
    if a.shape[-1] < fft_window:
        raise ValueError('at least one complete FFT frame required')
    if not np.isscalar(fs) or not np.isreal(fs) or not np.isfinite(fs) or fs <= 0:
        raise ValueError('positive finite sampling frequency required')
    if not np.isscalar(overlap) or not np.isreal(overlap) or not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('overlap must lie in [0, 1)')
    bands = np.asarray(frequency_bands, dtype=float)
    if bands.ndim != 2 or bands.shape[0] == 0 or bands.shape[1] != 2 or not np.all(np.isfinite(bands)):
        raise ValueError('finite nonempty band boundary pairs required')
    if np.any(bands[:, 0] < 0) or np.any(bands[:, 0] >= bands[:, 1]) or np.any(bands[:, 1] > fs / 2):
        raise ValueError('bands must satisfy 0 <= low < high <= Nyquist')
    frequencies = np.fft.fftfreq(fft_window, d=1. / fs)[:fft_window // 2]
    indices = [np.flatnonzero((frequencies >= low) & (frequencies < high)) for low, high in bands]
    if any(len(index) == 0 for index in indices):
        raise ValueError('every band must contain at least one FFT bin')
    taper = np.hanning(fft_window)
    norm = np.linalg.norm(taper) ** 2
    starts = range(0, a.shape[-1] - fft_window + 1, fft_window - int(overlap * fft_window))
    result = np.empty((len(a), len(bands), len(bands), a.shape[1]), dtype=float)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for wi, signal in enumerate(a):
            slices = np.empty((len(starts), a.shape[1], len(bands)), dtype=np.complex128)
            for si, start in enumerate(starts):
                transformed = np.fft.fft(taper * signal[:, start:start + fft_window]).T
                for bi, index in enumerate(indices):
                    slices[si, :, bi] = transformed[index].mean(0)
            powers = np.mean(abs(slices) ** 2, axis=0) / norm
            if not np.all(np.isfinite(powers)) or np.any(powers <= 0):
                raise ValueError('positive finite power required for every channel and band')
            for ci in range(a.shape[1]):
                cross = np.dot(slices[:, ci].T, np.conjugate(slices[:, ci])) / norm
                cross /= len(slices)
                result[wi, :, :, ci] = abs(cross) ** 2 / np.outer(powers[ci], powers[ci])
    if not np.all(np.isfinite(result)):
        raise ValueError('undefined frequency coherence')
    return result
