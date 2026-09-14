"""Conditional-OLS autoregressive coefficient standard errors per channel."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_channel_ar_standard_errors(windows: AbstractArray, order: int = 5, subsample: int = 4) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], windows.shape[1], order + 1), dtype='float64')


@register_atom(witness_channel_ar_standard_errors)
def channel_ar_standard_errors(windows: NDArray[np.float64], order: int = 5,
                                subsample: int = 4) -> NDArray[np.float64]:
    """Return (windows, channels, order+1) coefficient standard errors.

    Subsample each channel, regress samples p..N-1 on an intercept and lags 1..p
    using the Moore-Penrose pseudoinverse (rcond=1e-15). Output columns follow that
    same intercept/lag order. Residual variance is SSR/(N-2p-1), multiplying
    pinv(X) @ pinv(X).T before taking square roots of diagonal entries. This is
    the conditional-OLS ARResults.bse convention used by source ARError, not the
    AR prediction residuals or the maximum-likelihood residual-variance feature.

    Require finite real window/channel/sample tensors, positive integer p and
    subsample, N>2p+1 after subsampling, and full numerical column rank. Rank-
    deficient designs are rejected rather than returning ambiguous coefficients
    or the source wrapper's NaN fallback. Historical reference: statsmodels v0.8.0;
    original environment reproduction is not asserted. Inputs are not mutated.
    """
    a = np.asarray(windows)
    if a.dtype.kind not in 'iuf' or a.ndim != 3 or min(a.shape) == 0 or not np.all(np.isfinite(a)):
        raise ValueError('finite nonempty window/channel/sample tensor required')
    for value in [order, subsample]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('positive integer order and subsample required')
    result = np.empty((a.shape[0], a.shape[1], order + 1), dtype=float)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for wi, window in enumerate(a):
            for ci, channel in enumerate(window):
                values = channel[::subsample].astype(float)
                if len(values) <= 2 * order + 1:
                    raise ValueError('positive residual degrees of freedom required')
                target = values[order:]
                design = np.column_stack([np.ones(len(target))] +
                                         [values[order - lag:len(values) - lag] for lag in range(1, order + 1)])
                if np.linalg.matrix_rank(design) != order + 1:
                    raise ValueError('full-rank intercept/lag design required')
                inverse = np.linalg.pinv(design, rcond=1e-15)
                residual = target - design @ (inverse @ target)
                variance = np.dot(residual, residual) / (len(target) - order - 1)
                covariance = np.dot(inverse, inverse.T) * variance
                standard_errors = np.sqrt(np.diag(covariance))
                if not np.all(np.isfinite(standard_errors)):
                    raise ValueError('undefined AR coefficient standard errors')
                result[wi, ci] = standard_errors
    return result
