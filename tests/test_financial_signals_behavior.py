"""Behavioral tests for the financial_signals atom family."""

from __future__ import annotations

import numpy as np

from sciona.atoms.financial_signals.atoms import (
    book_imbalance_features,
    linear_trend_feature,
    realized_quadpower_quarticity,
    realized_volatility,
    weighted_average_price,
)


def test_realized_volatility_matches_closed_form() -> None:
    log_returns = np.array([0.1, -0.2, 0.05, 0.3], dtype=np.float64)
    expected = np.sqrt(np.sum(log_returns**2))
    assert realized_volatility(log_returns) == expected


def test_realized_quadpower_quarticity_uses_sparse_convolution_kernel() -> None:
    log_returns = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    expected = (np.pi**2 / 4.0) * 5.0 * (0.1 * 0.3 * 0.5)
    np.testing.assert_allclose(realized_quadpower_quarticity(log_returns), expected)


def test_realized_quadpower_quarticity_returns_zero_for_short_series() -> None:
    log_returns = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    assert realized_quadpower_quarticity(log_returns) == 0.0


def test_weighted_average_price_handles_zero_volume_rows() -> None:
    bid_price = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    ask_price = np.array([100.2, 101.2, 102.2], dtype=np.float64)
    bid_size = np.array([2.0, 0.0, 3.0], dtype=np.float64)
    ask_size = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    result = weighted_average_price(bid_price, ask_price, bid_size, ask_size)
    expected = np.array([(100.0 * 1.0 + 100.2 * 2.0) / 3.0, 0.0, (102.0 * 1.0 + 102.2 * 3.0) / 4.0])
    np.testing.assert_allclose(result, expected)


def test_linear_trend_feature_returns_slope_from_least_squares_fit() -> None:
    series = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float64)
    np.testing.assert_allclose(linear_trend_feature(series), 2.0)


def test_book_imbalance_features_matches_closed_form() -> None:
    bid_size = np.array([4.0, 3.0, 0.0], dtype=np.float64)
    ask_size = np.array([2.0, 3.0, 0.0], dtype=np.float64)
    result = book_imbalance_features(bid_size, ask_size)
    expected = np.array([1.0 / 3.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected)
