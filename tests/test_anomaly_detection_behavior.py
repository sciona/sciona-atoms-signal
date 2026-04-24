"""Behavioral tests for the anomaly_detection atom family."""

from __future__ import annotations

import numpy as np

from sciona.atoms.anomaly_detection.atoms import (
    matrix_profile_anomaly_score,
    multiscale_anomaly_aggregation,
)


def _series_with_spike(length: int = 64, spike_index: int = 36) -> np.ndarray:
    x = np.linspace(0.0, 4.0 * np.pi, length, dtype=np.float64)
    series = np.sin(x)
    series[spike_index : spike_index + 3] += 4.0
    return series


def test_matrix_profile_anomaly_score_is_normalized_per_subsequence() -> None:
    series = _series_with_spike()
    window_size = 8

    score = matrix_profile_anomaly_score(series, window_size)

    assert score.shape == (series.shape[0] - window_size + 1,)
    assert np.isfinite(score).all()
    assert np.all(score >= 0.0)
    assert np.all(score <= 1.0)


def test_matrix_profile_anomaly_score_peaks_near_spike() -> None:
    series = _series_with_spike()
    window_size = 8

    score = matrix_profile_anomaly_score(series, window_size)
    peak = int(np.argmax(score))

    assert 28 <= peak <= 38


def test_multiscale_anomaly_aggregation_returns_series_aligned_score() -> None:
    series = _series_with_spike()
    score = multiscale_anomaly_aggregation(series, [6, 8, 10])

    assert score.shape == series.shape
    assert np.isfinite(score).all()
    assert np.all(score >= 0.0)
    assert np.all(score <= 1.0)


def test_multiscale_anomaly_aggregation_peaks_near_spike() -> None:
    series = _series_with_spike()
    score = multiscale_anomaly_aggregation(series, [6, 8, 10])
    assert score[36] >= 0.95
    assert score[37] >= 0.85
