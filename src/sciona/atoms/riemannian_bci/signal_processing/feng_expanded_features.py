"""Source expanded spectral/correlation features for Feng's remaining models."""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import scale
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_feng_expanded_features(filtered_segment: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(filtered_segment.shape[0] // 20000, 384), dtype='float64')


@register_atom(witness_feng_expanded_features)
def feng_expanded_features(filtered_segment: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return windows/384 features from source 16-channel signals at 400 Hz.

    Requires finite samples/channels input comprising complete 50-second windows.
    Source clips have twelve windows. Converts to source float32 filter output.
    Feature order: channel-major six mean log10 FFT magnitudes plus standard
    deviation (112), time-correlation strict upper triangle then sorted absolute
    eigenvalues (136), frequency-correlation triangle then eigenvalues (136).
    Bands are [.1,4,8,12,30,70,180] Hz. No taper or power normalization.

    Correlation inputs are standardized ACROSS channels at each sample/band,
    then row correlations computed. Source all-zero rows receive a final-entry
    1e-5 perturbation. NaN/negative-infinity correlation values are zeroed before
    eigendecomposition. Raw spectral negative infinities remain for downstream
    source classifier cleanup. Preserves source np.linalg.eig magnitude sorting,
    including numerical roundoff behavior. No input mutation or clinical claim.
    """
    x = np.asarray(filtered_segment)
    if (x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[1] != 16
            or x.shape[0] < 20000 or x.shape[0] % 20000 or not np.all(np.isfinite(x))):
        raise ValueError('finite samples/16-channel input with complete 20000-sample windows required')
    with np.errstate(over='raise', invalid='raise'):
        x = x.astype(np.float32)
    bins = np.digitize(np.fft.rfftfreq(20000, 1/400), [.1, 4, 8, 12, 30, 70, 180])

    def correlations(values):
        values = scale(values, axis=0)
        for row in values:
            if np.all(row == 0):
                row[-1] += .00001
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.corrcoef(values)
        matrix[np.isneginf(matrix) | np.isnan(matrix)] = 0
        eigenvalues = np.abs(np.linalg.eig(matrix)[0])
        eigenvalues.sort()
        result = np.r_[matrix[np.triu_indices(16, 1)], eigenvalues]
        result[np.isneginf(result) | np.isnan(result)] = 0
        return result

    result = []
    for start in range(0, len(x), 20000):
        window = x[start:start+20000]
        spectral = np.empty((16, 7), dtype=float)
        for channel in range(16):
            with np.errstate(divide='ignore'):
                spectrum = np.log10(np.abs(np.fft.rfft(window[:, channel])))
            grouped = pd.DataFrame({'fft': spectrum, 'band': bins}).groupby('band').mean()
            spectral[channel, :6] = grouped['fft'].iloc[1:-1].to_numpy()
            spectral[channel, 6] = np.std(window[:, channel])
        frequency_input = spectral[:, :6].copy()
        frequency_input[np.isneginf(frequency_input)] = 0
        # Source creates float64 xcor before across-channel standardization.
        time = correlations(window.T.astype(float, order='C'))
        frequency = correlations(frequency_input)
        result.append(np.r_[spectral.ravel(), time, frequency])
    return np.stack(result)
