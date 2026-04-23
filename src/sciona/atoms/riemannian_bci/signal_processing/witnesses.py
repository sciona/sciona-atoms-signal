"""Ghost witnesses for the Riemannian BCI signal processing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_segment_max_aggregation(
    predictions: AbstractArray,
    window_size: int,
    stride: int,
    n_samples: int,
) -> AbstractArray:
    """Aggregated predictions have shape (n_samples, n_classes)."""
    n_classes = predictions.shape[1]
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64")


def witness_ranked_prediction_blend(
    predictions: AbstractArray,
    weights: AbstractArray,
) -> AbstractArray:
    """Blended prediction is a 1-D array of length n_samples."""
    n_samples = predictions.shape[1]
    return AbstractArray(shape=(n_samples,), dtype="float64")
