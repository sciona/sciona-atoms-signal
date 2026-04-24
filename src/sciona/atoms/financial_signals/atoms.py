"""Financial signal atoms adapted from Optiver order-book feature engineering.

These atoms re-express a small, reusable subset of the feature pipeline from the
Optiver realized-volatility solution. The quarticity atom is adapted from the
source's price-domain convolution heuristic to the requested log-return API.
"""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
import scipy.linalg

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_book_imbalance_features,
    witness_linear_trend_feature,
    witness_realized_quadpower_quarticity,
    witness_realized_volatility,
    witness_weighted_average_price,
)


_QUARTICITY_KERNEL = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
_FLOAT_TINY = np.finfo(np.float64).tiny


def _is_finite_array(values: NDArray[np.float64]) -> bool:
    """Return whether an ndarray contains only finite values."""
    return bool(np.isfinite(values).all())


def _same_shape(*arrays: NDArray[np.float64]) -> bool:
    """Return whether all arrays share the same shape."""
    return all(arr.shape == arrays[0].shape for arr in arrays[1:])


@register_atom(witness_realized_volatility)
@icontract.require(lambda log_returns: log_returns.ndim == 1, "log_returns must be 1-D")
@icontract.require(lambda log_returns: _is_finite_array(log_returns), "log_returns must be finite")
@icontract.ensure(lambda result: np.isfinite(result), "result must be finite")
@icontract.ensure(lambda result: result >= 0.0, "result must be non-negative")
def realized_volatility(log_returns: NDArray[np.float64]) -> float:
    """Compute realized volatility as sqrt(sum(log_returns^2))."""
    return float(np.sqrt(np.sum(np.square(log_returns, dtype=np.float64), dtype=np.float64)))


@register_atom(witness_realized_quadpower_quarticity)
@icontract.require(lambda log_returns: log_returns.ndim == 1, "log_returns must be 1-D")
@icontract.require(lambda log_returns: _is_finite_array(log_returns), "log_returns must be finite")
@icontract.ensure(lambda result: np.isfinite(result), "result must be finite")
@icontract.ensure(lambda result: result >= 0.0, "result must be non-negative")
def realized_quadpower_quarticity(log_returns: NDArray[np.float64]) -> float:
    """Compute a sparse-kernel quarticity proxy from absolute log returns.

    The upstream Optiver feature used a log-domain convolution over positive
    prices. For a signed log-return input we preserve the same vectorized
    structure by operating on absolute returns and a sparse kernel
    ``[1, 0, 1, 0, 1]``.
    """
    n_obs = int(log_returns.shape[0])
    if n_obs < _QUARTICITY_KERNEL.size:
        return 0.0

    stabilized = np.clip(np.abs(log_returns), _FLOAT_TINY, None)
    log_products = np.convolve(np.log(stabilized), _QUARTICITY_KERNEL, mode="valid")
    product_terms = np.exp(log_products)
    return float((np.pi**2 / 4.0) * n_obs * np.sum(product_terms, dtype=np.float64))


@register_atom(witness_weighted_average_price)
@icontract.require(lambda bid_price: bid_price.ndim >= 1, "bid_price must be at least 1-D")
@icontract.require(lambda bid_price, ask_price, bid_size, ask_size: _same_shape(bid_price, ask_price, bid_size, ask_size), "all inputs must share the same shape")
@icontract.require(lambda bid_price, ask_price, bid_size, ask_size: _is_finite_array(bid_price) and _is_finite_array(ask_price) and _is_finite_array(bid_size) and _is_finite_array(ask_size), "all inputs must be finite")
@icontract.ensure(lambda result, bid_price: result.shape == bid_price.shape, "result must preserve input shape")
@icontract.ensure(lambda result: _is_finite_array(result), "result must be finite")
def weighted_average_price(
    bid_price: NDArray[np.float64],
    ask_price: NDArray[np.float64],
    bid_size: NDArray[np.float64],
    ask_size: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the top-of-book weighted average price."""
    numerator = bid_price * ask_size + ask_price * bid_size
    denominator = bid_size + ask_size
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0.0,
    )


@register_atom(witness_linear_trend_feature)
@icontract.require(lambda series: series.ndim == 1, "series must be 1-D")
@icontract.require(lambda series: series.shape[0] >= 2, "series must contain at least two samples")
@icontract.require(lambda series: _is_finite_array(series), "series must be finite")
@icontract.ensure(lambda result: np.isfinite(result), "result must be finite")
def linear_trend_feature(series: NDArray[np.float64]) -> float:
    """Fit a least-squares line over the sample index and return its slope."""
    x = np.arange(series.shape[0], dtype=np.float64)
    design = np.column_stack((np.ones_like(x), x))
    coefs, _, _, _ = scipy.linalg.lstsq(design, series)
    return float(coefs[1])


@register_atom(witness_book_imbalance_features)
@icontract.require(lambda bid_size: bid_size.ndim >= 1, "bid_size must be at least 1-D")
@icontract.require(lambda bid_size, ask_size: bid_size.shape == ask_size.shape, "bid_size and ask_size must share the same shape")
@icontract.require(lambda bid_size, ask_size: _is_finite_array(bid_size) and _is_finite_array(ask_size), "inputs must be finite")
@icontract.ensure(lambda result, bid_size: result.shape == bid_size.shape, "result must preserve input shape")
@icontract.ensure(lambda result: _is_finite_array(result), "result must be finite")
def book_imbalance_features(
    bid_size: NDArray[np.float64],
    ask_size: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the signed bid-ask size imbalance."""
    denominator = bid_size + ask_size
    numerator = bid_size - ask_size
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0.0,
    )
