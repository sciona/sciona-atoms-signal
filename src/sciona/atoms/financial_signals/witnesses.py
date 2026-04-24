"""Ghost witnesses for the financial_signals atom family."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_realized_volatility(log_returns: AbstractArray) -> AbstractScalar:
    """Realized volatility collapses a return trace to a scalar."""
    return AbstractScalar(dtype="float64", min_val=0.0)


def witness_realized_quadpower_quarticity(log_returns: AbstractArray) -> AbstractScalar:
    """Quarticity summary collapses a return trace to a non-negative scalar."""
    return AbstractScalar(dtype="float64", min_val=0.0)


def witness_weighted_average_price(
    bid_price: AbstractArray,
    ask_price: AbstractArray,
    bid_size: AbstractArray,
    ask_size: AbstractArray,
) -> AbstractArray:
    """WAP preserves the order-book sample shape."""
    return AbstractArray(shape=bid_price.shape, dtype="float64")


def witness_linear_trend_feature(series: AbstractArray) -> AbstractScalar:
    """Least-squares slope collapses a 1-D trace to a scalar."""
    return AbstractScalar(dtype="float64")


def witness_book_imbalance_features(
    bid_size: AbstractArray,
    ask_size: AbstractArray,
) -> AbstractArray:
    """Book imbalance preserves the order-book sample shape."""
    return AbstractArray(shape=bid_size.shape, dtype="float64")
