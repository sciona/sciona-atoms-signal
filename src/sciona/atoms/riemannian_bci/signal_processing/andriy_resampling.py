"""Explicit documented FIR resampling for reconstructed Andriy feature paths."""
import math
import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin, resample_poly
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_documented_resample(filtered_segment: AbstractArray, sampling_frequency: int, target_frequency: int) -> AbstractArray:
    return AbstractArray(shape=(filtered_segment.shape[0], (filtered_segment.shape[1]*target_frequency+sampling_frequency-1)//sampling_frequency), dtype='float64')


@register_atom(witness_andriy_documented_resample)
def andriy_documented_resample(filtered_segment: NDArray[np.float64], sampling_frequency: int, target_frequency: int) -> NDArray[np.float64]:
    """Resample finite channels/samples to source target128Hz or256Hz.

    Explicit current MathWorks documented design: reduce p/q, use FIR order
    20*max(p,q), cutoff1/max(p,q) relative to Nyquist, Kaiser beta5, unit DC
    normalization before interpolation gainp. Polyphase processing compensates
    linear-phase delay and zero-pads outside the input, returning ceil(N*p/q)
    samples. Equal rates return an independent float64 copy.

    This is a documented-design reconstruction checked against Octave with
    identical explicit coefficients. Octave's DEFAULT resample filter differs;
    neither Octave-default nor historical MATLAB-default parity is claimed.
    Source Andriy preprocessing calls unspecified MATLAB resample defaults, so
    this distinction must remain in any downstream graph's review limitations.
    Source: https://www.mathworks.com/help/signal/ref/resample.html
    """
    for rate in [sampling_frequency, target_frequency]:
        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, np.integer)) or rate < 1:
            raise ValueError('positive integer sampling rates required')
    if target_frequency not in {128, 256}:
        raise ValueError('source target frequency must be128 or256Hz')
    x = np.asarray(filtered_segment)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or min(x.shape) == 0 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty real channels/samples required')
    divisor = math.gcd(int(sampling_frequency), int(target_frequency))
    p, q = int(target_frequency)//divisor, int(sampling_frequency)//divisor
    if p == q:
        return x.astype(float).copy()
    maximum = max(p, q)
    coefficients = firwin(20*maximum+1, 1/maximum, window=('kaiser', 5.))
    result = resample_poly(x.astype(float), p, q, axis=1, window=coefficients, padtype='constant')
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite resampled output')
    return result
