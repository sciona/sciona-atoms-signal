"""Source time-domain helper features for Andriy's mainstream feature family."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_time_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 10), dtype='float64')


@register_atom(witness_andriy_time_features)
def andriy_time_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ten source helper features for finite windows/samples matrices.

    Order: zero crossings, minmax, RMS, Hjorth activity/mobility/complexity,
    first/second derivative crossings, first/second derivative sample variance.
    Crossings require strictly negative adjacent products; touching zero does
    not count. Minmax equals first-derivative crossings, intentionally duplicated.
    Hjorth uses population variances; derivative variances use N-1. Preserve
    exact source epsilon placement: mobility=std(d1)/(std(x)+eps), complexity=
    (std(d2)/std(d1)+eps)/mobility. Undefined constant-signal ratios stay NaN/inf.
    At least four samples required; inputs are not mutated. The upstream source
    low-range all-feature masking is a separate operation, not applied here.
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 4 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty windows/samples with at least four samples required')
    result = []
    for row in x.astype(float):
        first, second = np.diff(row), np.diff(row, n=2)
        def crossings(a):
            return np.sum(a[:-1]*a[1:] < 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mobility = np.std(first)/(np.std(row)+np.finfo(float).eps)
            complexity = (np.std(second)/np.std(first)+np.finfo(float).eps)/mobility
        result.append([crossings(row), crossings(first), np.sqrt(row@row/len(row)),
                       np.var(row), mobility, complexity, crossings(first), crossings(second),
                       np.var(first, ddof=1), np.var(second, ddof=1)])
    return np.array(result, dtype=float)
