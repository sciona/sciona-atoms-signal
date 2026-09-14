"""Per-channel tangent features with explicit source reference-update semantics."""
import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from sklearn.covariance import shrunk_covariance
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _matrix_function(matrix, function, positive=True):
    values, vectors = scipy.linalg.eigh(matrix)
    if positive and np.any(values <= 0):
        raise ValueError('strictly positive eigenvalues required after shrinkage')
    return vectors @ np.diag(function(values)) @ vectors.T


def _checked_matrices(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or a.ndim != 4 or min(a.shape) == 0 or a.shape[1] != a.shape[2] or not np.all(np.isfinite(a)):
        raise ValueError('finite nonempty window/matrix/matrix/channel tensor required')
    a = a.astype(float)
    if not np.allclose(a, a.swapaxes(1, 2), rtol=0., atol=1e-12):
        raise ValueError('symmetric matrices required')
    result = np.empty_like(a)
    for ci in range(a.shape[-1]):
        for wi, matrix in enumerate(a[..., ci]):
            values = scipy.linalg.eigvalsh(matrix)
            if values.min() < -1e-12 * max(1., float(np.max(abs(values)))) or np.trace(matrix) <= 0:
                raise ValueError('positive semidefinite matrices with positive trace required')
            result[wi, :, :, ci] = shrunk_covariance(matrix, shrinkage=1e-9)
            if scipy.linalg.eigvalsh(result[wi, :, :, ci]).min() <= 0:
                raise ValueError('source shrinkage did not produce positive definite matrix')
    return result


def _reference(matrices, metric):
    if metric == 'identity':
        return np.eye(matrices.shape[1])
    average_log = np.zeros(matrices.shape[1:])
    for matrix in matrices:
        average_log += (1. / len(matrices)) * _matrix_function(matrix, np.log)
    return _matrix_function(average_log, np.exp, positive=False)


def _project(matrices, reference):
    inverse = _matrix_function(reference, lambda values: 1. / np.sqrt(values))
    size = matrices.shape[1]
    indices = np.triu_indices(size)
    weights = (np.sqrt(2.) * np.triu(np.ones((size, size)), 1) + np.eye(size))[indices]
    return np.stack([weights * _matrix_function(inverse @ matrix @ inverse, np.log)[indices]
                     for matrix in matrices])


def witness_channel_tangent_features(training_matrices: AbstractArray, prediction_matrices: AbstractArray,
                                     metric: str = 'logeuclid', tsupdate: bool = True) -> tuple[AbstractArray, AbstractArray]:
    p = training_matrices.shape[1]
    size = p * (p + 1) // 2 * training_matrices.shape[3]
    return (AbstractArray(shape=(training_matrices.shape[0], size), dtype='float64'),
            AbstractArray(shape=(prediction_matrices.shape[0], size), dtype='float64'))


@register_atom(witness_channel_tangent_features)
def channel_tangent_features(training_matrices: NDArray[np.float64], prediction_matrices: NDArray[np.float64],
                             metric: str = 'logeuclid', tsupdate: bool = True) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepare training and prediction features from per-channel PSD matrices.

    Both inputs have shape (windows, p, p, channels), with matching p/channels.
    Apply source Shrinkage(1e-9): (1-alpha)*C + alpha*trace(C)/p*I. For each
    channel fit an identity or log-Euclidean training reference, then project
    whitened matrix logarithms into weighted row-major upper triangles. Concatenate
    channels in input order to return two (windows, channels*p*(p+1)/2) arrays.

    tsupdate=True computes the prediction reference from the ENTIRE prediction
    batch. Results then depend on batch composition (source transductive behavior).
    False reuses the training reference. The source autocorrelation model uses
    identity/False; its frequency coherence model uses logeuclid/True. Only these
    two reference metrics are supported; neither is the iterative Riemannian mean.
    Input matrices may be singular, but must be PSD with positive trace and become
    strictly positive definite after the source's explicit fixed shrinkage.
    """
    if metric not in {'identity', 'logeuclid'} or not isinstance(tsupdate, bool):
        raise ValueError('identity/logeuclid metric and boolean tsupdate required')
    train = _checked_matrices(training_matrices)
    prediction = _checked_matrices(prediction_matrices)
    if train.shape[1:] != prediction.shape[1:]:
        raise ValueError('training and prediction matrix/channel dimensions must match')
    training_features, prediction_features = [], []
    for ci in range(train.shape[-1]):
        reference = _reference(train[..., ci], metric)
        training_features.append(_project(train[..., ci], reference))
        prediction_reference = _reference(prediction[..., ci], metric) if tsupdate else reference
        prediction_features.append(_project(prediction[..., ci], prediction_reference))
    result = np.concatenate(training_features, axis=1), np.concatenate(prediction_features, axis=1)
    if any(not np.all(np.isfinite(array)) for array in result):
        raise ValueError('nonfinite tangent features')
    return result
