"""Relative log power using the source's mean-per-band Welch convention."""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_relative_log_band_power(windows: AbstractArray, frequency_bands: list,
                                    fs: float = 400., fft_window: int = 512,
                                    overlap: float = .25) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], windows.shape[1], len(frequency_bands)), dtype='float64')


@register_atom(witness_relative_log_band_power)
def relative_log_band_power(windows: NDArray[np.float64], frequency_bands: list,
                            fs: float = 400., fft_window: int = 512,
                            overlap: float = .25) -> NDArray[np.float64]:
    """Return (windows, channels, bands) natural-log relative spectral powers.

    Welch uses a periodic Hann taper, per-frame constant detrending, one-sided
    density spectra, complete frames and arithmetic averaging. Average spectral
    density bins in each half-open band, then divide each band mean by the sum
    of band means for that channel before taking natural logs. Unequal band
    widths receive no integration weighting. This matches RelativeLogPower in
    the pinned competition source under current SciPy; it is not log of integrated
    band energy. Every band must contain a bin and have positive finite power.
    Input windows must be finite real (windows, channels, samples), with at least
    one complete FFT frame. Zero-power/undefined logarithms are rejected.
    """
    a = np.asarray(windows)
    if a.dtype.kind not in 'iuf' or a.ndim != 3 or min(a.shape) == 0 or not np.all(np.isfinite(a)):
        raise ValueError('finite nonempty window/channel/sample tensor required')
    if isinstance(fft_window, bool) or not isinstance(fft_window, (int, np.integer)) or fft_window < 3 or a.shape[-1] < fft_window:
        raise ValueError('integer FFT window >= 3 and at least one complete frame required')
    if not np.isscalar(fs) or not np.isreal(fs) or not np.isfinite(fs) or fs <= 0:
        raise ValueError('positive finite sampling frequency required')
    if not np.isscalar(overlap) or not np.isreal(overlap) or not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('overlap must lie in [0, 1)')
    bands = np.asarray(frequency_bands, dtype=float)
    if bands.ndim != 2 or bands.shape[0] == 0 or bands.shape[1] != 2 or not np.all(np.isfinite(bands)):
        raise ValueError('finite nonempty band boundary pairs required')
    if np.any(bands[:, 0] < 0) or np.any(bands[:, 0] >= bands[:, 1]) or np.any(bands[:, 1] > fs / 2):
        raise ValueError('bands must satisfy 0 <= low < high <= Nyquist')
    frequencies = np.fft.rfftfreq(fft_window, 1. / fs)
    masks = [(frequencies >= low) & (frequencies < high) for low, high in bands]
    if any(not mask.any() for mask in masks):
        raise ValueError('every band must contain a spectral bin')
    result = []
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for window in a.astype(float):
            _, power = welch(window, fs=fs, window='hann', nperseg=fft_window,
                             noverlap=int(fft_window * overlap), detrend='constant',
                             return_onesided=True, scaling='density', average='mean')
            means = np.array([power[:, mask].mean(1) for mask in masks])
            if not np.all(np.isfinite(means)) or np.any(means <= 0):
                raise ValueError('positive finite power required in every channel and band')
            result.append(np.log(means / np.sum(means, axis=0)).T)
    result = np.array(result)
    if not np.all(np.isfinite(result)):
        raise ValueError('undefined relative log power')
    return result
