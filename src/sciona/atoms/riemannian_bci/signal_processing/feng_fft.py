"""Source FFT feature tensor for the Feng base model branches."""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_feng_fft_features(filtered_segment: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(filtered_segment.shape[1], 7, filtered_segment.shape[0] // 12000), dtype='float64')


@register_atom(witness_feng_fft_features)
def feng_fft_features(filtered_segment: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return (channels, seven features, complete 30-second windows) at 400 Hz.

    Input is finite real samples/channels, converted to source filter-output
    float32. Six features are means of log10(abs(rFFT)) within half-open bands
    [0.1,4), [4,8), [8,12), [12,30), [30,70), [70,180) Hz. DC and frequencies
    at or above 180 Hz are excluded. No taper, detrending, power conversion or
    energy normalization is applied. The seventh feature is population standard
    deviation of the raw window. Source pandas grouped means are retained.
    Windows do not overlap; trailing samples are discarded. Source segments have
    twenty windows. Zero FFT magnitudes can yield negative infinity/NaN; these
    intentionally remain for the source classifier's explicit cleanup stage.
    This numerical feature operation does not itself produce probabilities.
    """
    x = np.asarray(filtered_segment)
    if (x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] < 12000
            or x.shape[1] == 0 or not np.all(np.isfinite(x))):
        raise ValueError('finite samples/channels with a complete 12000-sample window required')
    with np.errstate(over='raise', invalid='raise'):
        x = x.astype(np.float32)
    n_windows = len(x) // 12000
    result = np.empty((x.shape[1], 7, n_windows), dtype=np.float64)
    bins = np.digitize(np.fft.rfftfreq(12000, d=1./400), [.1, 4, 8, 12, 30, 70, 180])
    for channel in range(x.shape[1]):
        for frame in range(n_windows):
            window = x[frame*12000:(frame+1)*12000, channel]
            with np.errstate(divide='ignore'):
                spectrum = np.log10(np.abs(np.fft.rfft(window)))
            means = pd.DataFrame({'fft': spectrum, 'band': bins}).groupby('band').mean()
            result[channel, :6, frame] = means['fft'].iloc[1:-1].to_numpy()
            result[channel, 6, frame] = np.std(window)
    return result
