"""Ordered source-duration preprocessing for Feng training/prediction partitions."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.registry import register_atom
from .feng_filter import feng_resample_filter
from .feng_fft import feng_fft_features


def witness_feng_filter_partitions(training_segments, prediction_segments):
    return training_segments, prediction_segments


@register_atom(witness_feng_filter_partitions)
def feng_filter_partitions(training_segments: list, prediction_segments: list) -> tuple[list, list]:
    """Filter each ordered samples/channels segment as a 600-second source clip.

    Both partitions must be nonempty lists with a common channel count. Input
    sample counts may differ: Fourier resampling uses the source's fixed duration
    and 400 Hz output rate. Callers must supply actual 600-second clips. Each
    segment receives independent zero-state causal filtering. Labels and order
    are untouched. Delegates to the exact source-compatible filtering provider.
    """
    if not isinstance(training_segments, list) or not isinstance(prediction_segments, list) or not training_segments or not prediction_segments:
        raise ValueError('two nonempty ordered segment lists required')
    segments = training_segments + prediction_segments
    arrays = [np.asarray(x) for x in segments]
    if any(x.ndim != 2 for x in arrays) or len({x.shape[1] for x in arrays}) != 1:
        raise ValueError('samples/channels segments with a common channel count required')
    filtered = [feng_resample_filter(x, 600) for x in arrays]
    return filtered[:len(training_segments)], filtered[len(training_segments):]


def witness_feng_feature_partitions(training_filtered, prediction_filtered):
    from sciona.ghost.abstract import AbstractArray
    return (AbstractArray(shape=(len(training_filtered), training_filtered[0].shape[1], 7, 20), dtype='float64'),
            AbstractArray(shape=(len(prediction_filtered), prediction_filtered[0].shape[1], 7, 20), dtype='float64'))


@register_atom(witness_feng_feature_partitions)
def feng_feature_partitions(training_filtered: list, prediction_filtered: list) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stack source FFT tensors in segment order after fixed-duration filtering.

    Both partitions require 240000-sample clips with matching channel count.
    Outputs have axes segments/channels/seven features/twenty windows. Source
    undefined log features remain for the classifier's explicit cleanup stage.
    """
    if not isinstance(training_filtered, list) or not isinstance(prediction_filtered, list) or not training_filtered or not prediction_filtered:
        raise ValueError('two nonempty ordered filtered segment lists required')
    arrays = [np.asarray(x) for x in training_filtered + prediction_filtered]
    if any(x.ndim != 2 or x.shape[0] != 240000 for x in arrays) or len({x.shape[1] for x in arrays}) != 1:
        raise ValueError('source-duration filtered segments with matching channels required')
    features = [feng_fft_features(x) for x in arrays]
    n = len(training_filtered)
    return np.stack(features[:n]), np.stack(features[n:])
