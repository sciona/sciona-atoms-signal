"""Andriy AR fit features with explicit least-squares initial-state semantics."""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .andriy_ar_coefficients import andriy_ar_coefficients


def _estimated_prediction(data, polynomial):
    """Project prefix residuals onto the observable initial-history subspace.

    A pure AR one-step predictor has a nilpotent shift state transition; its
    initial state affects at most p outputs. Use an SVD output-space projection
    with cutoff p*eps*smax to avoid constructing potentially enormous initial
    histories. This is an explicit numerical contract, not a reproduction of
    historical MATLAB findstates rank decisions or state coordinates.
    """
    order=len(polynomial)-1
    baseline=lfilter(np.r_[0.,-polynomial[1:]], [1.], data)
    observability=np.zeros((order,order))
    for time in range(order):
        observability[time,:order-time]=-polynomial[time+1:]
    left,singular,_=np.linalg.svd(observability,full_matrices=False)
    rank=np.count_nonzero(singular>order*np.finfo(float).eps*singular[0])
    basis=left[:,:rank]
    baseline[:order]+=basis@(basis.T@(data[:order]-baseline[:order]))
    return baseline


def witness_andriy_documented_ar_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 9), dtype='float64')


@register_atom(witness_andriy_documented_ar_features)
def andriy_documented_ar_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return nine source-order AR fit scores with documented initial-state loss.

    Input: finite windows/samples with at least 20 samples. Fit each polynomial
    on the first floor(N/2) samples and predict the final floor(N/2), omitting
    an odd middle sample. Whole-window range<.01 yields NaN. Otherwise first-half
    range<1e-5 yields 50; second-half range<1e-5 yields 100, in that precedence.
    Scores are 100*(1-error_norm/centered_test_norm), without clipping.

    Initial conditions minimize squared one-step prediction error using the
    explicit SVD rank convention in _estimated_prediction. This follows the
    documented loss, not verified historical MATLAB toolbox numerical behavior.
    Finite inputs exclude the source's NaN/inf repair branch; low-range masking
    skips source random replacement without consuming global RNG. No mutation.
    """
    x=np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim!=2 or x.shape[0]==0 or x.shape[1]<20 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty windows/samples with at least twenty samples required')
    result=np.full((len(x),9),np.nan)
    for row_index,row in enumerate(x.astype(float)):
        if np.ptp(row)<.01:
            continue
        half=len(row)//2
        model,test=row[:half],row[-half:]
        if np.ptp(model)<1e-5:
            result[row_index]=50
        elif np.ptp(test)<1e-5:
            result[row_index]=100
        else:
            polynomials=andriy_ar_coefficients(model[None,:])[0]
            denominator=np.linalg.norm(test-np.mean(test))
            for order in range(1,10):
                predicted=_estimated_prediction(test,polynomials[order-1,:order+1])
                result[row_index,order-1]=100*(1-np.linalg.norm(predicted-test)/denominator)
    return result
