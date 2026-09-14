"""Source-compatible resampling and causal filtering for Feng model branches."""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, resample
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_feng_resample_filter(segment: AbstractArray, duration_seconds: int = 600) -> AbstractArray:
    return AbstractArray(shape=(400 * duration_seconds, segment.shape[1]), dtype='float32')


@register_atom(witness_feng_resample_filter)
def feng_resample_filter(segment: NDArray[np.float64], duration_seconds: int = 600) -> NDArray[np.float32]:
    """Return samples/channels resampled to 400 Hz and causally bandpass filtered.

    Reproduces pinned Feng process_file_filter/filter: cast raw input to float32,
    Fourier-resample along samples to 400*duration_seconds, then fifth-order
    Butterworth bandpass [0.1,180] Hz with direct-form lfilter and zero initial
    state, finally cast to float32. Source duration is 600 seconds; shorter
    positive integer durations support explicit equivalent synthetic workloads.
    The caller supplies the actual duration; sample count alone does not encode
    an input sampling rate. FFT resampling assumes periodic extension. This is
    causal filtering, not zero-phase filtering; source edge transients remain.
    Finite nonempty real samples/channels are required. Inputs are not mutated.
    """
    a = np.asarray(segment)
    if (a.ndim != 2 or min(a.shape) == 0 or a.shape[0] < 2
            or a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a))):
        raise ValueError('finite real samples/channels with at least two samples required')
    if isinstance(duration_seconds, (bool, np.bool_)) or not isinstance(duration_seconds, (int, np.integer)) or duration_seconds < 1:
        raise ValueError('positive integer duration_seconds required')
    with np.errstate(over='raise', invalid='raise'):
        sampled = resample(a.astype(np.float32), 400 * int(duration_seconds), axis=0)
        b, coefficients = butter(5, np.array([0.1, 180.]) / 200., btype='band')
        result = lfilter(b, coefficients, sampled, axis=0).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError('source filtering produced nonfinite values')
    return result
