"""Ordered per-channel summary features for the source combined-feature model."""
import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_channel_basic_statistics(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], windows.shape[1], 6), dtype='float64')


@register_atom(witness_channel_basic_statistics)
def channel_basic_statistics(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return (windows, channels, 6) source BasicStats features in fixed order.

    Columns: arithmetic mean, population standard deviation (ddof=0), biased
    Fisher excess kurtosis, biased skewness, 90th percentile, 10th percentile.
    Percentiles use linear interpolation. Inputs must be finite nonempty real
    window/channel/sample tensors with at least two samples and nonconstant
    channels; undefined or nonfinite moments are rejected. No input mutation.
    These are descriptive features, not unbiased population estimates or a
    clinical classifier. Source: pinned competition preproc.BasicStats.
    """
    a = np.asarray(windows)
    if a.dtype.kind not in 'iuf' or a.ndim != 3 or min(a.shape) == 0 or a.shape[-1] < 2 or not np.all(np.isfinite(a)):
        raise ValueError('finite nonempty window/channel/sample tensor with at least two samples required')
    result = []
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for window in a.astype(float):
            sd = np.std(window, axis=1, ddof=0)
            if np.any(sd == 0):
                raise ValueError('nonconstant channels required for standardized moments')
            features = np.c_[np.mean(window, axis=1), sd,
                             kurtosis(window, axis=1, fisher=True, bias=True),
                             skew(window, axis=1, bias=True),
                             np.percentile(window, 90, axis=1, method='linear'),
                             np.percentile(window, 10, axis=1, method='linear')]
            if not np.all(np.isfinite(features)):
                raise ValueError('undefined or nonfinite channel statistics')
            result.append(features)
    return np.array(result)
