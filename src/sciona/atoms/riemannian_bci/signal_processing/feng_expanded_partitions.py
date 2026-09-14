"""Ordered full-duration expanded-feature tensors for the two Feng models."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .feng_expanded_features import feng_expanded_features


def witness_feng_expanded_partitions(training_filtered, prediction_filtered):
    return (AbstractArray(shape=(len(training_filtered), 12, 384), dtype='float64'),
            AbstractArray(shape=(len(prediction_filtered), 12, 384), dtype='float64'))


@register_atom(witness_feng_expanded_partitions)
def feng_expanded_partitions(training_filtered: list, prediction_filtered: list) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stack source expanded features in training/prediction segment order.

    Requires two nonempty lists of finite 240000-sample, 16-channel filtered
    segments. Outputs are segments/twelve windows/384 features. Numerical
    extraction delegates to the registered source-expanded feature provider;
    raw spectral undefined values remain for each classifier's distinct cleanup.
    """
    if not isinstance(training_filtered, list) or not isinstance(prediction_filtered, list) or not training_filtered or not prediction_filtered:
        raise ValueError('two nonempty ordered filtered segment lists required')
    arrays = [np.asarray(x) for x in training_filtered + prediction_filtered]
    if any(x.shape != (240000, 16) for x in arrays):
        raise ValueError('source-duration 240000-sample/16-channel filtered segments required')
    features = [feng_expanded_features(x) for x in arrays]
    n = len(training_filtered)
    return np.stack(features[:n]), np.stack(features[n:])
