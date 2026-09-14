"""Numerical multi-order AR coefficient stage of the Andriy source branch."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_ar_coefficients(model_windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(model_windows.shape[0], 9, 10), dtype='float64')


@register_atom(witness_andriy_ar_coefficients)
def andriy_ar_coefficients(model_windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit orders 1..9 using the source zero-padded QR regression convention.

    Input: finite model-windows/samples, at least ten samples. Output has shape
    windows/9/10: each order's monic A polynomial, followed by zero padding.
    Uncentered inputs are padded with nine zeros at each end. Lower-order fits
    use the next regressor column then reverse/negate coefficients; order nine
    uses the response column. SVD pseudoinverses use dimension*eps cutoff.
    This is only the numerical regression stage: no idpoly construction,
    initial-state prediction, sentinel handling, or nine-feature fit scores.
    No input mutation. Current full QR agrees with source chunked QR to tolerance.
    """
    x=np.asarray(model_windows)
    if x.dtype.kind not in 'iuf' or x.ndim!=2 or x.shape[0]==0 or x.shape[1]<10 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty model-windows/samples with at least ten samples required')
    result=np.zeros((len(x),9,10),dtype=float)
    result[:,:,0]=1
    for row_index,row in enumerate(x.astype(float)):
        padded=np.pad(row,(9,9))
        times=np.arange(9,len(padded))
        regression=np.column_stack([-padded[times-lag] for lag in range(1,10)]+[padded[times]])
        triangular=np.linalg.qr(regression,mode='r')
        for order in range(1,10):
            coefficients=np.linalg.pinv(triangular[:order,:order],rtol=order*np.finfo(float).eps)@triangular[:order,order]
            if order<9:
                coefficients=-coefficients[::-1]
            result[row_index,order-1,1:order+1]=coefficients
    return result
