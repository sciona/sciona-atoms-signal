"""Behavioral tests for riemannian_bci signal_processing atoms."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.riemannian_bci.signal_processing.atoms import (
    ranked_prediction_blend,
    segment_max_aggregation,
)


# -- segment_max_aggregation --------------------------------------------------


def test_segment_max_aggregation_shape() -> None:
    predictions = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]])
    result = segment_max_aggregation(predictions, window_size=4, stride=2, n_samples=10)
    assert result.shape == (10, 2)


def test_segment_max_aggregation_non_overlapping() -> None:
    """Non-overlapping windows should directly assign predictions."""
    predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = segment_max_aggregation(predictions, window_size=3, stride=3, n_samples=6)
    np.testing.assert_allclose(result[0], [1.0, 2.0])
    np.testing.assert_allclose(result[1], [1.0, 2.0])
    np.testing.assert_allclose(result[2], [1.0, 2.0])
    np.testing.assert_allclose(result[3], [3.0, 4.0])
    np.testing.assert_allclose(result[4], [3.0, 4.0])
    np.testing.assert_allclose(result[5], [3.0, 4.0])


def test_segment_max_aggregation_overlapping_takes_max() -> None:
    """Overlapping region should take max of assigned predictions."""
    predictions = np.array([[1.0], [3.0]])
    # Window 0 covers [0, 1, 2], Window 1 covers [1, 2, 3]
    result = segment_max_aggregation(predictions, window_size=3, stride=1, n_samples=4)
    # Sample 0: only window 0 -> 1.0
    assert result[0, 0] == 1.0
    # Sample 1: max(1.0, 3.0) -> 3.0
    assert result[1, 0] == 3.0
    # Sample 2: max(1.0, 3.0) -> 3.0
    assert result[2, 0] == 3.0
    # Sample 3: only window 1 -> 3.0
    assert result[3, 0] == 3.0


def test_segment_max_aggregation_uncovered_samples_zero() -> None:
    """Samples not covered by any window should be zero."""
    predictions = np.array([[1.0, 2.0]])
    result = segment_max_aggregation(predictions, window_size=2, stride=1, n_samples=5)
    # Only samples 0 and 1 are covered
    np.testing.assert_allclose(result[0], [1.0, 2.0])
    np.testing.assert_allclose(result[1], [1.0, 2.0])
    np.testing.assert_allclose(result[2], [0.0, 0.0])
    np.testing.assert_allclose(result[3], [0.0, 0.0])
    np.testing.assert_allclose(result[4], [0.0, 0.0])


# -- ranked_prediction_blend -------------------------------------------------


def test_ranked_prediction_blend_shape() -> None:
    predictions = np.array([[0.1, 0.4, 0.2], [0.3, 0.5, 0.1]])
    weights = np.array([1.0, 1.0])
    result = ranked_prediction_blend(predictions, weights)
    assert result.shape == (3,)


def test_ranked_prediction_blend_equal_weights_preserves_order() -> None:
    """With identical predictions across models, rank should be consistent."""
    predictions = np.array([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
    weights = np.array([0.5, 0.5])
    result = ranked_prediction_blend(predictions, weights)
    # Original ranking: 1 -> rank 1, 2 -> rank 2, 3 -> rank 3
    assert result[0] < result[2] < result[1]


def test_ranked_prediction_blend_single_model() -> None:
    """With one model, output ranks should match input ranks."""
    predictions = np.array([[10.0, 30.0, 20.0]])
    weights = np.array([1.0])
    result = ranked_prediction_blend(predictions, weights)
    expected_ranks = np.array([1.0, 3.0, 2.0])
    np.testing.assert_allclose(result, expected_ranks)


def test_ranked_prediction_blend_output_is_rank() -> None:
    """Output should consist of valid rank values."""
    rng = np.random.default_rng(50)
    predictions = rng.standard_normal((3, 20))
    weights = np.array([1.0, 2.0, 1.0])
    result = ranked_prediction_blend(predictions, weights)
    assert result.min() >= 1.0
    assert result.max() <= 20.0
    # Sum of ranks 1..n = n*(n+1)/2
    np.testing.assert_allclose(result.sum(), 20 * 21 / 2)
