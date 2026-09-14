"""Andriy 180-feature connectivity extractor with explicit numerical settings."""
from itertools import combinations
import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, correlate, csd, filtfilt, hilbert, welch
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


_BANDS = ((.5, 4), (3, 8), (7, 15), (14, 31), (30, 100))
_PAIR_INDICES = (1,4,5,16,18,19,20,30,32,33,34,45,46,55,58,59,66,68,69,70,
                 76,78,79,80,87,88,93,94,96,97,102,103,104,106,108,109,110,113,114,115,118,120)
_ALL_PAIRS = tuple(combinations(range(16), 2))
_PAIRS = tuple(_ALL_PAIRS[i-1] for i in _PAIR_INDICES)
_MONTAGES = (
    ((1,2,3,4,9,10,11,12), (5,6,7,8,13,14,15,16)),
    ((1,5,9,13,3,7,11,15), (2,6,10,14,4,8,12,16)),
    (tuple(range(1,13)), tuple(range(5,17))),
    ((1,5,9,13,2,6,9,14,3,7,11,15), (2,6,10,14,3,7,11,15,4,8,12,16)),
    ((1,2,3,5,6,7,9,10,11), (6,7,8,10,11,12,14,15,16)),
    ((2,3,4,6,7,8,10,11,12), (5,6,7,9,10,11,13,14,15)),
)
_GROUPS = tuple(tuple(_PAIRS.index((a-1,b-1)) for a,b in zip(left,right))
                for left,right in _MONTAGES)


def witness_andriy_connectivity_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 180), dtype='float64')


def _correlation(left, right):
    left, right = left-np.mean(left), right-np.mean(right)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sum(left*right)/np.sqrt(np.sum(left**2)*np.sum(right**2))


@register_atom(witness_andriy_connectivity_features)
def andriy_connectivity_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return 180 features from finite windows/16 channels/samples at 256 Hz.

    Require at least 255 samples. Welch/CPSD use symmetric 255-point Hamming,
    zero overlap, 256-point FFT, no detrending. Five bands use separate fifth
    order lowpass then highpass Butterworth filters, forward/reverse filtering
    with odd reflection of length 15. Current SciPy filter arithmetic is used;
    historical MATLAB numerical identity is not claimed.

    Preserve 42 source pairs, six source montages (including repeated pairs),
    and final montage coherence's source reuse of montage five. Output order is
    montage, metric (lag, asynchrony, symmetry, peak coherence frequency, mean
    coherence, envelope correlation), band. NaN metric values are omitted from
    montage averages. Entirely zero windows yield NaN as in the caller gate.
    Partial channel NaNs are outside this finite-input contract. No mutation.
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 3 or x.shape[0] == 0 or x.shape[1] != 16 or x.shape[2] < 255 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty windows/16 channels/samples with at least 255 samples required')
    result = np.full((len(x), 180), np.nan)
    settings = dict(fs=256., window=np.hamming(255), nperseg=255, noverlap=0,
                    nfft=256, detrend=False, scaling='density')
    for row_index, segment in enumerate(x.astype(float)):
        if not np.any(segment):
            continue
        frequency, psd = welch(segment, axis=1, **settings)
        coherence = []
        for left,right in _PAIRS:
            _, cross = csd(segment[left], segment[right], **settings)
            with np.errstate(divide='ignore', invalid='ignore'):
                coherence.append(np.abs(cross)**2/(psd[left]*psd[right]))
        coherence = np.asarray(coherence)
        # metric, band, pair
        metrics = np.full((6, 5, len(_PAIRS)), np.nan)
        for band_index,(lo,hi) in enumerate(_BANDS):
            selected = (frequency >= lo) & (frequency <= hi)
            low_b,low_a = butter(5, hi/128, btype='low')
            high_b,high_a = butter(5, lo/128, btype='high')
            filtered = filtfilt(high_b, high_a,
                               filtfilt(low_b, low_a, segment, axis=1, padlen=15),
                               axis=1, padlen=15)
            envelope = np.abs(hilbert(filtered, axis=1))**2
            for pair_index,(left,right) in enumerate(_PAIRS):
                left_psd, right_psd = psd[left,selected]*129, psd[right,selected]*129
                with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    symmetry = 1-np.nanmean(np.abs((left_psd-right_psd)/(left_psd+right_psd)))
                    metrics[2,band_index,pair_index] = symmetry
                    metrics[1,band_index,pair_index] = abs(_correlation(left_psd,right_psd))-symmetry
                    metrics[4,band_index,pair_index] = np.nanmean(coherence[pair_index,selected])
                metrics[3,band_index,pair_index] = frequency[selected][np.argmax(coherence[pair_index,selected])]
                metrics[5,band_index,pair_index] = _correlation(envelope[left],envelope[right])
                cross = correlate(filtered[left], filtered[right], mode='full', method='fft')
                center = len(segment[0])-1
                metrics[0,band_index,pair_index] = np.argmax(np.abs(cross[center-128:center+129]))-128
        features = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for group_index,group in enumerate(_GROUPS):
                for metric in range(6):
                    indices = _GROUPS[4] if group_index == 5 and metric == 4 else group
                    features.extend(np.nanmean(metrics[metric][:,indices], axis=1))
        result[row_index] = features
    return result
