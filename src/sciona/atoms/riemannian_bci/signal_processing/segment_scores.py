"""Segment-level reduction and normalized rank blending with explicit identities."""
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_segment_probability_max(window_probabilities: AbstractArray, segment_indices: AbstractArray,
                                     n_segments: int) -> AbstractArray:
    return AbstractArray(shape=(n_segments,), dtype='float64')


@register_atom(witness_segment_probability_max)
def segment_probability_max(window_probabilities: NDArray[np.float64], segment_indices: NDArray[np.int64],
                             n_segments: int) -> NDArray[np.float64]:
    """Maximum window probability per zero-based segment, returned in segment order.

    Equivalent to the source's groupby(segment).max() after explicit ID alignment.
    Every segment must have at least one window; no missing segments are invented.
    Windows need not be sorted and segments may have different window counts.
    """
    p, ids = np.asarray(window_probabilities), np.asarray(segment_indices)
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 1:
        raise ValueError('positive integer segment count required')
    if p.dtype.kind not in 'iuf' or p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('finite probability vector required')
    if ids.dtype.kind not in 'iu' or ids.shape != p.shape or not np.array_equal(np.unique(ids), np.arange(n_segments)):
        raise ValueError('indices must cover all zero-based segments')
    result = np.full(n_segments, -np.inf)
    np.maximum.at(result, ids, p)
    return result


def witness_normalized_rank_blend(predictions: AbstractArray, weights: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(predictions.shape[1],), dtype='float64')


@register_atom(witness_normalized_rank_blend)
def normalized_rank_blend(predictions: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weighted mean of per-model average ranks divided by the segment count.

    Rows are models; columns must already share the same segment identity/order.
    Ties get average ranks. Normalize nonnegative weights to sum to one. Do not
    re-rank the resulting mean: the source make_blend.py writes it directly.
    The complete source ensemble has eleven equally weighted model inputs.
    Other model counts/weights are explicit variants, not that full ensemble.
    """
    p, w = np.asarray(predictions), np.asarray(weights)
    if p.dtype.kind not in 'iuf' or p.ndim != 2 or min(p.shape) == 0 or not np.all(np.isfinite(p)):
        raise ValueError('finite nonempty model/segment matrix required')
    if w.dtype.kind not in 'iuf' or w.shape != (len(p),) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError('matching nonnegative weights with positive finite sum required')
    result = np.zeros(p.shape[1])
    for row, weight in zip(p, w / w.sum()):
        result += weight * (rankdata(row) / len(row))
    return result
