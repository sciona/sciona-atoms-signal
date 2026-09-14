"""Andriy source centering, causal filtering and inclusive edge trimming."""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_filter_trim(segment: AbstractArray, sampling_frequency: int) -> AbstractArray:
    return AbstractArray(shape=(segment.shape[0], segment.shape[1]-8*sampling_frequency+1), dtype='float64')


@register_atom(witness_andriy_filter_trim)
def andriy_filter_trim(segment: NDArray[np.float64], sampling_frequency: int) -> NDArray[np.float64]:
    """Center channels, apply source notch/high-pass recurrence, and trim edges.

    Input is finite real channels/samples at integer sampling_frequency >120Hz,
    with at least eight seconds of samples. Per-channel centering precedes a
    zero-state 60Hz notch with conjugate unit-circle zeros and poles scaled .95.
    High-pass recurrence is h[0]=notch[0], then
    h[t]=(1-pi/fs)*h[t-1]+notch[t]-notch[t-1], in that arithmetic order.
    Source MATLAB slice 4*fs:end-4*fs is inclusive, translated to Python
    [4*fs-1:n-4*fs]. Thus n-8*fs+1 samples remain. This preserves the source's
    asymmetric one-sample endpoint; it does not perform resampling to128/256Hz.
    Float64 execution, independent output, no input mutation. Formula-level
    reconstruction; MATLAB engine parity remains unproven.
    """
    x = np.asarray(segment)
    if isinstance(sampling_frequency, (bool, np.bool_)) or not isinstance(sampling_frequency, (int, np.integer)) or sampling_frequency <= 120:
        raise ValueError('integer sampling frequency above120Hz required')
    fs = int(sampling_frequency)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 8*fs or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty channels/samples with at least eight seconds required')
    x = x.astype(float)
    zeros = np.exp(np.array([1j, -1j])*np.pi*(60/(fs/2)))
    b, a = np.poly(zeros), np.poly(.95*zeros)
    centered = x-x.mean(axis=1, keepdims=True)
    notch = lfilter(b, a, centered, axis=1)
    highpass = np.empty_like(notch)
    highpass[:, 0] = notch[:, 0]
    alpha = 2*np.pi*.5/fs
    for t in range(1, x.shape[1]):
        highpass[:, t] = (1-alpha)*highpass[:, t-1]+notch[:, t]-notch[:, t-1]
    result = highpass[:, 4*fs-1:x.shape[1]-4*fs].copy()
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite source filter output')
    return result
