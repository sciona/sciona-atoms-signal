"""Source within-clip imputation and MATLAB-order feature assembly."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_assemble_clip_features(mainstream, autoregressive, csp_autoregressive, connectivity):
    return (AbstractArray(shape=(mainstream.shape[1], 1965), dtype='float64'),
            AbstractArray(shape=(mainstream.shape[1],), dtype='bool'))


@register_atom(witness_andriy_assemble_clip_features)
def andriy_assemble_clip_features(mainstream: NDArray[np.float64], autoregressive: NDArray[np.float64],
                                  csp_autoregressive: NDArray[np.float64], connectivity: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Impute one clip and return ordered windows/1965 features plus validity.

    Shapes: mainstream (16,windows,102), AR (16,windows,9), CSP AR
    (components,windows,9), connectivity (windows,180). Source uses only the first
    CSP component. NaNs are filled with the within-clip mean of the same channel
    and feature across windows; wholly missing series stay NaN. Next mainstream
    feature indices 21:52 and 89:95 (MATLAB one-based inclusive) receive log(1+x).
    Values below -1 in those features are rejected because the source would
    become complex; -1 produces source negative infinity, not a validity failure.

    Assembly uses feature-major/channel-minor MATLAB flattening, then AR in the
    same order, first CSP component, connectivity. Validity is source non-NaN
    testing after transforms, NOT finite testing. All rows are retained here:
    callers discard invalid training rows but retain prediction rows and mask.
    Inputs may contain NaN but not infinity. Returns independent float64 arrays.
    Formula reconstruction; MATLAB runtime parity remains to be established.
    """
    arrays = [np.asarray(v) for v in [mainstream, autoregressive, csp_autoregressive, connectivity]]
    if any(a.dtype.kind not in 'iuf' or np.any(np.isinf(a)) for a in arrays):
        raise ValueError('real feature tensors without infinities required; NaN allowed')
    main, ar, csp, con = arrays
    if main.ndim != 3 or main.shape[0] != 16 or main.shape[2] != 102 or main.shape[1] < 1:
        raise ValueError('mainstream shape must be 16/windows/102')
    windows = main.shape[1]
    if ar.shape != (16, windows, 9) or csp.ndim != 3 or csp.shape[0] < 1 or csp.shape[1:] != (windows, 9) or con.shape != (windows, 180):
        raise ValueError('AR, CSP and connectivity axes must match the source feature contract')

    def impute(value, axis):
        result = value.astype(float).copy()
        present = ~np.isnan(result)
        count = present.sum(axis=axis, keepdims=True)
        total = np.where(present, result, 0).sum(axis=axis, keepdims=True)
        mean = np.full_like(total, np.nan)
        np.divide(total, count, out=mean, where=count != 0)
        return np.where(present, result, mean)

    main, ar = impute(main, 1), impute(ar, 1)
    csp, con = impute(csp[0], 0), impute(con, 0)
    indices = np.r_[20:52, 88:95]
    selected = main[:, :, indices]
    if np.any(selected < -1):
        raise ValueError('selected logarithmic features must be at least -1 or NaN')
    with np.errstate(divide='ignore', invalid='ignore'):
        # Preserve source log(1+x), not the numerically different log1p(x).
        main[:, :, indices] = np.log(1+selected)
    result = np.concatenate([main.transpose(1, 2, 0).reshape(windows, 1632),
                             ar.transpose(1, 2, 0).reshape(windows, 144), csp, con], axis=1)
    return result, ~np.isnan(result).any(axis=1)
