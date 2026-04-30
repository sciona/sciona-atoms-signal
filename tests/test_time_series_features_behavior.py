"""Behavioral tests for time-series feature atoms."""

from __future__ import annotations

import numpy as np

from sciona.atoms.time_series_features import (
    create_lag_features,
    entity_time_aggregate,
    exogenous_feature_concat,
    expm1_inverse,
    forward_fill,
    grouped_temporal_diff,
    log1p_transform,
    rolling_window_features,
    seasonal_decompose_additive,
    technical_indicators_macd,
)


def test_rolling_window_features_matches_hand_computed_values() -> None:
    series = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = rolling_window_features(series, 3, ["mean", "max"])

    expected = np.array(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [2.0, 3.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_create_lag_features_is_causal_without_wraparound() -> None:
    values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    result = create_lag_features(values, [1, 2])

    expected = np.array(
        [
            [np.nan, np.nan],
            [10.0, np.nan],
            [20.0, 10.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_technical_indicators_macd_returns_aligned_components() -> None:
    prices = np.array([10.0, 11.0, 12.0, 11.0, 13.0], dtype=np.float64)
    macd, signal, histogram = technical_indicators_macd(prices, fast=2, slow=3, signal=2)

    assert macd.shape == prices.shape
    assert signal.shape == prices.shape
    assert histogram.shape == prices.shape
    np.testing.assert_allclose(histogram, macd - signal)
    assert np.all(np.isfinite(macd))


def test_seasonal_decompose_additive_reconstructs_observed_series_where_defined() -> None:
    series = np.array([10.0, 20.0, 12.0, 22.0, 14.0, 24.0], dtype=np.float64)
    trend, seasonal, residual = seasonal_decompose_additive(series, period=2)

    assert trend.shape == series.shape
    assert seasonal.shape == series.shape
    assert residual.shape == series.shape
    reconstructed = trend + seasonal + residual
    np.testing.assert_allclose(reconstructed[~np.isnan(trend)], series[~np.isnan(trend)])


def test_grouped_temporal_diff_does_not_cross_entity_boundaries() -> None:
    values = np.array([1.0, 3.0, 10.0, 15.0, 21.0], dtype=np.float64)
    entities = np.array([1, 1, 2, 2, 2], dtype=np.int64)
    result = grouped_temporal_diff(values, entities, periods=1)

    expected = np.array([np.nan, 2.0, np.nan, 5.0, 6.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_entity_time_aggregate_returns_sorted_entity_summaries() -> None:
    values = np.array([1.0, 3.0, 10.0, 20.0], dtype=np.float64)
    entities = np.array([2, 1, 2, 1], dtype=np.int64)
    result = entity_time_aggregate(values, entities, ["mean", "max", "count"])

    expected = np.array([[11.5, 20.0, 2.0], [5.5, 10.0, 2.0]], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_forward_fill_propagates_last_valid_value_by_column() -> None:
    values = np.array(
        [
            [np.nan, 1.0],
            [2.0, np.nan],
            [np.nan, 3.0],
            [4.0, np.nan],
        ],
        dtype=np.float64,
    )
    result = forward_fill(values, axis=0)

    expected = np.array(
        [
            [np.nan, 1.0],
            [2.0, 1.0],
            [2.0, 3.0],
            [4.0, 3.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_log1p_transform_round_trips_with_expm1_inverse() -> None:
    values = np.array([0.0, 1.0, 10.0, 1000.0], dtype=np.float64)
    transformed = log1p_transform(values)
    np.testing.assert_allclose(expm1_inverse(transformed), values)


def test_exogenous_feature_concat_broadcasts_static_values() -> None:
    sequence = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    exogenous = np.array([9.0, 10.0], dtype=np.float64)
    result = exogenous_feature_concat(sequence, exogenous)

    expected = np.array(
        [[1.0, 2.0, 9.0, 10.0], [3.0, 4.0, 9.0, 10.0], [5.0, 6.0, 9.0, 10.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(result, expected)
