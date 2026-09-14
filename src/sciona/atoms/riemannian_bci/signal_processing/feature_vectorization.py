"""Fit-shape-preserving vectorization for window feature tensors."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_vectorize_feature_partitions(training_tensor: AbstractArray, prediction_tensor: AbstractArray) -> tuple:
    size = 1
    for dimension in training_tensor.shape[1:]:
        size *= dimension
    return (AbstractArray(shape=(training_tensor.shape[0], size), dtype='float64'),
            AbstractArray(shape=(prediction_tensor.shape[0], size), dtype='float64'))


@register_atom(witness_vectorize_feature_partitions)
def vectorize_feature_partitions(training_tensor: NDArray[np.float64], prediction_tensor: NDArray[np.float64]) -> tuple:
    """Return training/prediction matrices flattened after their sample axes.

    Preserve C-order feature ordering and require matching non-sample shapes,
    reproducing MNE Vectorizer.fit_transform(train), then transform(prediction).
    For (windows, channels, bands), bands vary fastest, then channels. Equal
    flattened size alone is insufficient: the original channel/band shape must
    agree. Inputs must be finite numeric tensors of at least two dimensions with
    nonempty axes. Outputs are independent float64 copies; no fitted model needs
    persistence or deserialization. The historical reference is MNE v0.13.
    """
    train, prediction = np.asarray(training_tensor), np.asarray(prediction_tensor)
    for tensor in [train, prediction]:
        if tensor.dtype.kind not in 'iuf' or tensor.ndim < 2 or min(tensor.shape) == 0 or not np.all(np.isfinite(tensor)):
            raise ValueError('finite nonempty numeric feature tensors of at least two dimensions required')
    if train.shape[1:] != prediction.shape[1:]:
        raise ValueError('training and prediction feature shapes must match')
    return (train.reshape(len(train), -1).astype(float, copy=True),
            prediction.reshape(len(prediction), -1).astype(float, copy=True))
