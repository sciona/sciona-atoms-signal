"""The competition's exact PFD, two-scale HFD and expanding-range Hurst features."""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _pfd(values):
    signs = np.sign(np.diff(values))
    changes = np.count_nonzero(np.roll(signs, 1) != signs)
    n = len(values)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + .4 * changes)))


def _hfd(values):
    n = len(values)
    lengths = [np.sum(abs(np.diff(values)))]
    scales = []
    for offset in [1, 2]:
        steps = (n - offset) // 2
        indices = np.arange(offset - 1, offset + (steps + 1) * 2 - 1, 2)
        scales.append(np.sum(abs(np.diff(values[indices]))) * (n - 1.) / (steps * 2))
    lengths.append(np.mean(scales))
    if min(lengths) <= 0:
        raise ValueError('positive path lengths at both source HFD scales required')
    design = np.column_stack([np.log(1. / np.arange(1., 3.)), np.ones(2)])
    return np.linalg.lstsq(design, np.log(lengths), rcond=None)[0][0]


def _hurst(values):
    centered = values - values.mean()
    cumulative = np.cumsum(centered)
    ranges = (np.maximum.accumulate(cumulative) - np.minimum.accumulate(cumulative))[1:]
    deviations = pd.Series(centered).expanding().std(ddof=1).to_numpy()[1:]
    deviations[deviations == 0] = 1e-12
    ranges += 1e-12
    dependent = np.log(ranges / deviations)
    design = np.column_stack([np.log(np.arange(1, len(dependent) + 1)), np.ones(len(dependent))])
    return np.linalg.lstsq(design, dependent, rcond=None)[0][0]


def witness_channel_fractal_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], windows.shape[1], 3), dtype='float64')


@register_atom(witness_channel_fractal_features)
def channel_fractal_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return source VariousFeatures in PFD, HFD, Hurst column order.

    Inputs are finite nonempty (windows, channels, samples) tensors, at least four
    samples per channel. PFD counts cyclic changes in derivative signs, including
    zero signs. HFD uses only k=1,2 and the source's path-length normalization;
    it omits the extra 1/k factor used by some other Higuchi implementations, so
    no standard fractal-dimension range is asserted. Hurst fits log expanding
    cumulative range/sample deviation against log(1..N-1), using the source's
    1e-12 range offset and zero-deviation replacement. No clipping or alternative
    estimator is substituted. Require positive path lengths at both HFD scales.

    Preserves source numerical output while computing Hurst on a copy instead
    of mutating the caller's channel. Uses current pandas expanding sample std
    and NumPy least squares; not a historical-library or physiological-validity
    claim. Zero-scale, nonfinite and undefined feature regimes are rejected.
    """
    a = np.asarray(windows)
    if a.dtype.kind not in 'iuf' or a.ndim != 3 or min(a.shape) == 0 or a.shape[-1] < 4 or not np.all(np.isfinite(a)):
        raise ValueError('finite nonempty window/channel/sample tensor with at least four samples required')
    result = np.empty((a.shape[0], a.shape[1], 3), dtype=float)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for wi, window in enumerate(a):
            for ci, channel in enumerate(window):
                values = channel.astype(float)
                result[wi, ci] = [_pfd(values), _hfd(values), _hurst(values)]
    if not np.all(np.isfinite(result)):
        raise ValueError('undefined fractal feature')
    return result
