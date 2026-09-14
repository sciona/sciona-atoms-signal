"""Distribution and SVD features from the Andriy competition feature family."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_distribution_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 4), dtype='float64')


@register_atom(witness_andriy_distribution_features)
def andriy_distribution_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return kurtosis, skewness, SVD entropy, and Fisher information per window.

    Moments use population variance without bias correction; kurtosis includes
    the Gaussian baseline of three. The source SVD matrix has 20 rows and N-20
    columns, intentionally omitting the final input sample. Singular values
    are normalized by their sum, with no epsilon, zero clipping, or special
    handling of degenerate inputs. Undefined results remain NaN/inf for the
    separate upstream low-range mask. Accept finite windows/samples matrices
    with at least 21 samples. Inputs are not mutated.
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 21 or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty windows/samples with at least 21 samples required')
    result = []
    for row in x.astype(float):
        centered = row - np.mean(row)
        variance = np.mean(centered**2)
        embedding = np.lib.stride_tricks.sliding_window_view(row, 20)[:-1].T
        singular = np.linalg.svd(embedding, compute_uv=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            kurtosis = np.mean(centered**4) / variance**2
            skewness = np.mean(centered**3) / variance**1.5
            weights = singular / np.sum(singular)
            entropy = -np.sum(weights * np.log2(weights))
            fisher = np.sum(np.diff(weights)**2 / weights[:-1])
        result.append([kurtosis, skewness, entropy, fisher])
    return np.asarray(result, dtype=float)
