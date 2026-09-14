"""Explicit population and identity adapters for the eleven-model graph."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_population_indices() -> tuple:
    return (0, 1, 2)


@register_atom(witness_population_indices)
def ensemble_population_indices() -> tuple[int, int, int]:
    """Return the three source population positions without inferring identities."""
    return (0, 1, 2)


def _groups(*values):
    if any(not isinstance(value, list) or len(value) != 3 for value in values):
        raise ValueError('exactly three ordered population entries required')


def _ids(value, count):
    array = np.asarray(value)
    if array.dtype.kind not in 'iu' or array.shape != (count,) or len(np.unique(array)) != count or count == 0:
        raise ValueError('unique nonempty integer identities matching prediction clips required')
    if any(int(value) > np.iinfo(np.int64).max or int(value) < np.iinfo(np.int64).min for value in array):
        raise ValueError('prediction identities must fit signed int64')
    return array.astype(np.int64)


def witness_select_ensemble_population(training_populations: list, prediction_populations: list,
        label_populations: list, identity_populations: list, population_index: int) -> tuple:
    _groups(training_populations, prediction_populations, label_populations, identity_populations)
    return (training_populations[population_index], prediction_populations[population_index],
            label_populations[population_index], identity_populations[population_index])


@register_atom(witness_select_ensemble_population)
def select_ensemble_population(training_populations: list, prediction_populations: list,
        label_populations: list, identity_populations: list, population_index: int) -> tuple[list, list, NDArray[np.int64], NDArray[np.int64]]:
    """Select one family's explicit population inputs, preserving segment order.

    Exactly three populations, each with nonempty training/prediction clip lists,
    binary per-clip labels including both classes, and one unique opaque integer
    ID per prediction clip, fitting signed int64. IDs must also be unique across populations. Clip
    orientation, sampling rate and source-safe membership are family contracts
    supplied by the caller, not inferred here. Alex uses channels/samples at
    400 Hz; Feng uses samples/channels full 600-second clips. The selected clip
    objects are passed through unchanged. All populations are checked before
    selection so wrong list lengths cannot silently misalign later scores.
    """
    _groups(training_populations, prediction_populations, label_populations, identity_populations)
    if isinstance(population_index, (bool, np.bool_)) or not isinstance(population_index, (int, np.integer)) or not 0 <= population_index < 3:
        raise ValueError('population index must be 0, 1 or 2')
    all_ids = []
    for train, pred, labels, ids in zip(training_populations, prediction_populations, label_populations, identity_populations):
        if not isinstance(train, list) or not isinstance(pred, list) or not train or not pred:
            raise ValueError('nonempty ordered clip lists required')
        labels = np.asarray(labels)
        if labels.dtype.kind not in 'iuf' or labels.shape != (len(train),) or not np.array_equal(np.unique(labels), [0, 1]):
            raise ValueError('both binary training classes with one label per clip required')
        all_ids.extend(map(int, _ids(ids, len(pred))))
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('prediction identities must be unique across populations')
    i = int(population_index)
    return training_populations[i], prediction_populations[i], np.asarray(label_populations[i], dtype=np.int64), np.asarray(identity_populations[i], dtype=np.int64)


def witness_collect_population_scores(scores0: AbstractArray, ids0: AbstractArray,
        scores1: AbstractArray, ids1: AbstractArray, scores2: AbstractArray, ids2: AbstractArray) -> tuple:
    count = sum(value.shape[0] for value in [scores0, scores1, scores2])
    return AbstractArray(shape=(count,), dtype='float64'), AbstractArray(shape=(count,), dtype='int64')


@register_atom(witness_collect_population_scores)
def collect_population_scores(scores0: NDArray[np.float64], ids0: NDArray[np.int64],
        scores1: NDArray[np.float64], ids1: NDArray[np.int64],
        scores2: NDArray[np.float64], ids2: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Concatenate three population score/ID pairs without ranking or sorting.

    Every finite score vector has one unique integer ID per entry. Population
    order is 0/1/2 and IDs must be globally unique; this preserves the source's
    population concatenation before final all-population rank blending.
    """
    scores, identities = [], []
    for value, ids in [(scores0, ids0), (scores1, ids1), (scores2, ids2)]:
        value = np.asarray(value)
        if value.dtype.kind not in 'iuf' or value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError('finite population score vectors required')
        identities.extend(map(int, _ids(ids, len(value))))
        scores.append(value)
    if len(set(identities)) != len(identities):
        raise ValueError('prediction identities must be unique across populations')
    return np.concatenate(scores).astype(float), np.asarray(identities, dtype=np.int64)


def witness_ordered_prediction_ids(prediction_populations: list, identity_populations: list) -> AbstractArray:
    _groups(prediction_populations, identity_populations)
    return AbstractArray(shape=(sum(map(len, prediction_populations)),), dtype='int64')


@register_atom(witness_ordered_prediction_ids)
def ordered_prediction_ids(prediction_populations: list, identity_populations: list) -> NDArray[np.int64]:
    """Bind Andriy concatenated output IDs to each raw population's clip count."""
    _groups(prediction_populations, identity_populations)
    ids = []
    for clips, values in zip(prediction_populations, identity_populations):
        if not isinstance(clips, list):
            raise ValueError('ordered prediction clip lists required')
        ids.extend(map(int, _ids(values, len(clips))))
    if len(set(ids)) != len(ids):
        raise ValueError('prediction identities must be unique across populations')
    return np.asarray(ids, dtype=np.int64)
