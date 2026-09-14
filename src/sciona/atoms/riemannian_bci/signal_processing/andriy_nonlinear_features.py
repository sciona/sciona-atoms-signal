"""Nonlinear energy, curve length, and amplitude entropy source features."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_nonlinear_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 3), dtype='float64')


@register_atom(witness_andriy_nonlinear_features)
def andriy_nonlinear_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return nonlinear energy, total curve length, and amplitude entropy.

    Entropy follows the source default histogram: ceil(sqrt(N)) equal cells,
    bounds expanded by half of range/(N-1), internal boundary ties assigned to
    the upper bin. Return differential entropy in nats with the source
    (cells-1)/(2*N) correction, including empty cells. Constant windows raise
    as the source histogram does; upstream random replacement and low-range
    masking are separate. Inputs must be finite windows/samples, N>=3.
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 3 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty windows/samples with at least three samples required')
    result = []
    for row in x.astype(float):
        low, high = np.min(row), np.max(row)
        if high <= low:
            raise ValueError('source histogram requires nonconstant windows')
        delta = (high-low)/(len(row)-1)
        lower, upper = low-delta/2, high+delta/2
        cells = int(np.ceil(np.sqrt(len(row))))
        # MATLAB round uses ties away from zero; all transformed values here
        # are positive. floor(value + .5) preserves its boundary convention.
        indices = np.floor((row-lower)/(upper-lower)*cells+1).astype(int)-1
        counts = np.bincount(indices, minlength=cells)
        weighted_log = 0.
        for count in counts:
            if count:
                weighted_log -= count*np.log(count)
        entropy = weighted_log/len(row)+np.log(len(row))+np.log((upper-lower)/cells)+(cells-1)/(2*len(row))
        energy = np.mean(row[1:-1]**2-row[:-2]*row[2:])
        result.append([energy, np.sum(np.abs(np.diff(row))), entropy])
    return np.asarray(result, dtype=float)
