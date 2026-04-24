"""Ghost witnesses for the anomaly_detection atom family."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_matrix_profile_anomaly_score(
    series: AbstractArray,
    window_size: int,
) -> AbstractArray:
    """Single-scale matrix-profile scores are defined per subsequence."""
    score_len = series.shape[0] - window_size + 1
    return AbstractArray(shape=(score_len,), dtype="float64")


def witness_multiscale_anomaly_aggregation(
    series: AbstractArray,
    window_sizes: list[int],
) -> AbstractArray:
    """Multi-scale aggregation lifts scores back to the original series length."""
    return AbstractArray(shape=series.shape, dtype="float64")
