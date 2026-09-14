"""CSP fitting and first-component projection for the Andriy source branch."""
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_csp_filters(preictal: AbstractArray, interictal: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(16, 16), dtype='float64')


def _matrix(value, name):
    x = np.asarray(value)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] != 16 or x.shape[1] < 16 or not np.all(np.isfinite(x)):
        raise ValueError(f'{name} must be finite 16 channels/samples with at least 16 samples')
    return x.astype(float)


def _covariance(x):
    covariance = x@x.T
    energy = np.trace(covariance)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError('each CSP class must have finite positive energy')
    covariance /= energy
    values = np.linalg.eigvalsh(covariance)
    if values[0] <= 1e-12*values[-1]:
        raise ValueError('CSP class covariance is singular or numerically ill-conditioned')
    return covariance


def _require_distinct(values):
    if np.any(np.diff(np.sort(values)) <= 64*np.finfo(float).eps*np.max(np.abs(values))):
        raise ValueError('CSP eigenvalues are not numerically distinct; source basis is ambiguous')


@register_atom(witness_andriy_csp_filters)
def andriy_csp_filters(preictal: NDArray[np.float64], interictal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit sixteen source-order CSP filters from two preprocessed class matrices.

    Uses trace-normalized uncentered second moments, descending composite
    eigendecomposition and ascending generalized class eigenvalues. Generalized
    eigenvectors have maximum absolute component one, matching the audited
    current Octave path. Filter signs are canonicalized to positive maximum
    loading; source numerical agreement is up to sign, not bitwise identity.
    Historical MATLAB scaling and downstream AR sign invariance remain unproven.

    Reject class covariance condition ratios <=1e-12 and indistinguishable
    composite/generalized eigenvalues instead of choosing an unstable basis.
    Caller clip selection, filtering/resampling and class concatenation are
    separate. No mean subtraction or regularization is introduced here.
    """
    first = _covariance(_matrix(preictal, 'preictal'))
    second = _covariance(_matrix(interictal, 'interictal'))
    values, vectors = np.linalg.eigh(first+second)
    _require_distinct(values)
    whitening = (1/np.sqrt(values[::-1]))[:,None]*vectors[:,::-1].T
    values, vectors = eig(whitening@first@whitening.T, whitening@second@whitening.T)
    if not np.all(np.isfinite(values)) or np.max(np.abs(values.imag)) > 1e-10 or np.max(np.abs(vectors.imag)) > 1e-10:
        raise ValueError('CSP generalized eigendecomposition is not finite and real')
    _require_distinct(values.real)
    vectors = vectors[:,np.argsort(values.real)].real
    vectors /= np.max(np.abs(vectors),axis=0)
    filters = vectors.T@whitening
    signs = np.sign(filters[np.arange(16), np.argmax(np.abs(filters),axis=1)])
    return filters*signs[:,None]


def witness_andriy_csp_project(segment: AbstractArray, filters: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(1, segment.shape[1]), dtype='float64')


@register_atom(witness_andriy_csp_project)
def andriy_csp_project(segment: NDArray[np.float64], filters: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the first filter as source spatFilt(...,1), yielding 1/samples.

    Accept a finite 16-channel matrix with at least 16 samples and a finite
    16-by-16 filter matrix. The first component is the smallest generalized
    class eigenvalue, not the highest-variance component. Do not refit filters.
    """
    x = _matrix(segment, 'segment')
    coefficients = np.asarray(filters)
    if coefficients.dtype.kind not in 'iuf' or coefficients.shape != (16,16) or not np.all(np.isfinite(coefficients)):
        raise ValueError('finite 16-by-16 CSP filters required')
    return coefficients.astype(float)[:1]@x
