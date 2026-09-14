"""Identity-aligned final eleven-model source blend with explicit baseline."""
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_aligned_eleven_model_blend(predictions: list, prediction_ids: list,
        output_ids: AbstractArray, baseline: AbstractArray) -> AbstractArray:
    if len(predictions) != 11 or len(prediction_ids) != 11:
        raise ValueError('exactly eleven model predictions and identity vectors required')
    return AbstractArray(shape=output_ids.shape, dtype='float64')


@register_atom(witness_aligned_eleven_model_blend)
def aligned_eleven_model_blend(predictions: list, prediction_ids: list,
        output_ids: NDArray[np.int64], baseline: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply source make_blend identity alignment and eleven equal rank weights.

    Supply eleven finite score vectors and corresponding unique integer opaque
    identity vectors in source model order: combined, autocorrelation, coherence,
    relative power, Feng XGB, Feng KNN, Feng expanded KNN, Feng GLM, Andriy SVM,
    Andriy GLM, Andriy XGB. Caller maps source identities to consistent opaque
    integers across branches; no source identifiers or mappings are persisted.

    Output IDs are unique and nonempty, and every model must contain every
    requested ID. Extra prediction IDs are allowed: rank over each complete
    model vector before selecting output IDs, preserving source denominators.
    Ties use average ranks. Add each normalized rank with weight 1/11 to the
    caller-supplied finite baseline in output-ID order. A zero baseline yields
    the usual rank blend; source code does not reset the supplied baseline.
    Do not re-rank or clip the sum. All inputs are unchanged. This only blends
    supplied predictions; it does not train or validate the eleven branches.
    """
    if not isinstance(predictions, list) or not isinstance(prediction_ids, list) or len(predictions) != 11 or len(prediction_ids) != 11:
        raise ValueError('exactly eleven model predictions and identity vectors required')
    ids, initial = np.asarray(output_ids), np.asarray(baseline)
    if ids.dtype.kind not in 'iu' or ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError('unique nonempty integer output IDs required')
    if initial.dtype.kind not in 'iuf' or initial.shape != ids.shape or not np.all(np.isfinite(initial)):
        raise ValueError('finite baseline matching output IDs required')
    prepared = []
    for scores, identity in zip(predictions, prediction_ids):
        scores, identity = np.asarray(scores), np.asarray(identity)
        if scores.dtype.kind not in 'iuf' or scores.ndim != 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
            raise ValueError('finite nonempty model score vectors required')
        if identity.dtype.kind not in 'iu' or identity.shape != scores.shape or len(np.unique(identity)) != len(identity):
            raise ValueError('unique integer model IDs matching scores required')
        positions = {int(value): index for index, value in enumerate(identity)}
        if any(int(value) not in positions for value in ids):
            raise ValueError('every model must cover every requested output ID')
        prepared.append((scores, np.array([positions[int(value)] for value in ids])))
    result = initial.astype(float, copy=True)
    for scores, positions in prepared:
        result += (1. / 11.) * (rankdata(scores) / len(scores))[positions]
    return result
