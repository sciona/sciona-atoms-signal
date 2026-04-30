"""Pure NumPy feature engineering atoms for aligned time series."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_create_lag_features,
    witness_entity_time_aggregate,
    witness_exogenous_feature_concat,
    witness_expm1_inverse,
    witness_forward_fill,
    witness_grouped_temporal_diff,
    witness_log1p_transform,
    witness_rolling_window_features,
    witness_seasonal_decompose_additive,
    witness_technical_indicators_macd,
)


_ROLLING_STATS = {"mean", "std", "min", "max"}
_AGGREGATIONS = {"mean", "std", "min", "max", "count"}


def _numeric_1d(values: NDArray[np.float64], allow_nan: bool = True) -> bool:
    array = np.asarray(values, dtype=np.float64)
    finite_or_nan = np.isfinite(array) | np.isnan(array)
    if not allow_nan:
        finite_or_nan = np.isfinite(array)
    return bool(array.ndim == 1 and array.size >= 1 and np.all(finite_or_nan))


def _numeric_2d(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _valid_stats(statistics: list[str]) -> bool:
    return bool(len(statistics) >= 1 and all(stat in _ROLLING_STATS for stat in statistics))


def _valid_lags(lags: list[int]) -> bool:
    return bool(len(lags) >= 1 and all(isinstance(lag, int) and lag > 0 for lag in lags))


def _valid_aggregations(agg_fns: list[str]) -> bool:
    return bool(len(agg_fns) >= 1 and all(name in _AGGREGATIONS for name in agg_fns))


def _same_1d_length(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    return bool(np.asarray(left).ndim == 1 and np.asarray(right).ndim == 1 and np.asarray(left).shape == np.asarray(right).shape)


def _finite_or_nan(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(array) | np.isnan(array)))


def _ema(prices: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    values = np.asarray(prices, dtype=np.float64)
    alpha = 2.0 / (float(span) + 1.0)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for idx in range(1, values.size):
        result[idx] = alpha * values[idx] + (1.0 - alpha) * result[idx - 1]
    return result


@register_atom(witness_rolling_window_features)
@icontract.require(lambda series: _numeric_1d(series), "series must be a 1-D numeric array")
@icontract.require(lambda window_size: isinstance(window_size, int) and window_size > 0, "window_size must be a positive integer")
@icontract.require(lambda statistics: _valid_stats(statistics), "statistics must be selected from mean/std/min/max")
@icontract.ensure(lambda result, series, statistics: result.shape == (np.asarray(series).shape[0], len(statistics)), "result must preserve time length and one column per statistic")
@icontract.ensure(lambda result: _finite_or_nan(result), "rolling features must be finite or NaN")
def rolling_window_features(
    series: NDArray[np.float64],
    window_size: int,
    statistics: list[str],
) -> NDArray[np.float64]:
    """Compute same-length rolling summary features with leading NaN padding."""
    values = np.asarray(series, dtype=np.float64)
    result = np.full((values.size, len(statistics)), np.nan, dtype=np.float64)
    if int(window_size) > values.size:
        return result

    windows = np.lib.stride_tricks.sliding_window_view(values, int(window_size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        columns = []
        for stat in statistics:
            if stat == "mean":
                columns.append(np.nanmean(windows, axis=1))
            elif stat == "std":
                columns.append(np.nanstd(windows, axis=1))
            elif stat == "min":
                columns.append(np.nanmin(windows, axis=1))
            else:
                columns.append(np.nanmax(windows, axis=1))
    result[int(window_size) - 1 :, :] = np.column_stack(columns)
    return result


@register_atom(witness_create_lag_features)
@icontract.require(lambda values: _numeric_1d(values), "values must be a 1-D numeric array")
@icontract.require(lambda lags: _valid_lags(lags), "lags must be positive integers")
@icontract.ensure(lambda result, values, lags: result.shape == (np.asarray(values).shape[0], len(lags)), "lag matrix shape must match time length and lag count")
@icontract.ensure(lambda result: _finite_or_nan(result), "lag features must be finite or NaN")
def create_lag_features(
    values: NDArray[np.float64],
    lags: list[int],
) -> NDArray[np.float64]:
    """Create strictly causal lag columns without circular wraparound."""
    series = np.asarray(values, dtype=np.float64)
    features = np.full((series.size, len(lags)), np.nan, dtype=np.float64)
    for col, lag in enumerate(lags):
        if lag < series.size:
            features[lag:, col] = series[:-lag]
    return features


@register_atom(witness_technical_indicators_macd)
@icontract.require(lambda prices: _numeric_1d(prices, allow_nan=False), "prices must be a finite 1-D numeric array")
@icontract.require(lambda prices: np.asarray(prices).shape[0] >= 2, "prices must contain at least two samples")
@icontract.require(lambda fast, slow, signal: isinstance(fast, int) and isinstance(slow, int) and isinstance(signal, int) and slow > fast > 0 and signal > 0, "MACD spans must satisfy slow > fast > 0 and signal > 0")
@icontract.ensure(lambda result, prices: all(part.shape == np.asarray(prices).shape for part in result), "MACD outputs must align to input length")
@icontract.ensure(lambda result: all(np.all(np.isfinite(part)) for part in result), "MACD outputs must be finite")
def technical_indicators_macd(
    prices: NDArray[np.float64],
    fast: int,
    slow: int,
    signal: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute MACD, signal line, and histogram from exponential averages."""
    values = np.asarray(prices, dtype=np.float64)
    macd = _ema(values, int(fast)) - _ema(values, int(slow))
    signal_line = _ema(macd, int(signal))
    histogram = macd - signal_line
    return macd, signal_line, histogram


@register_atom(witness_seasonal_decompose_additive)
@icontract.require(lambda series: _numeric_1d(series, allow_nan=False), "series must be a finite 1-D numeric array")
@icontract.require(lambda series, period: isinstance(period, int) and period > 1 and np.asarray(series).shape[0] >= 2 * period, "period must fit at least two full cycles")
@icontract.ensure(lambda result, series: all(part.shape == np.asarray(series).shape for part in result), "decomposition outputs must align to input length")
@icontract.ensure(lambda result: all(_finite_or_nan(part) for part in result), "decomposition outputs must be finite or NaN")
def seasonal_decompose_additive(
    series: NDArray[np.float64],
    period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split a series into additive trend, seasonal, and residual components."""
    values = np.asarray(series, dtype=np.float64)
    trend = np.full(values.shape, np.nan, dtype=np.float64)
    half = int(period) // 2
    for center in range(values.size):
        start = center - half
        stop = start + int(period)
        if start >= 0 and stop <= values.size:
            trend[center] = float(np.mean(values[start:stop]))

    detrended = values - trend
    seasonal_pattern = np.empty(int(period), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for offset in range(int(period)):
            seasonal_pattern[offset] = np.nanmean(detrended[offset:: int(period)])
    seasonal_pattern = seasonal_pattern - np.nanmean(seasonal_pattern)
    seasonal = np.resize(seasonal_pattern, values.size).astype(np.float64)
    residual = values - trend - seasonal
    return trend, seasonal, residual


@register_atom(witness_grouped_temporal_diff)
@icontract.require(lambda values: _numeric_1d(values, allow_nan=False), "values must be a finite 1-D numeric array")
@icontract.require(lambda values, entity_ids: _same_1d_length(values, entity_ids), "values and entity_ids must align")
@icontract.require(lambda periods: isinstance(periods, int) and periods > 0, "periods must be a positive integer")
@icontract.ensure(lambda result, values: result.shape == np.asarray(values).shape, "diff output must align to input length")
@icontract.ensure(lambda result: _finite_or_nan(result), "diff output must be finite or NaN")
def grouped_temporal_diff(
    values: NDArray[np.float64],
    entity_ids: NDArray[np.int64],
    periods: int,
) -> NDArray[np.float64]:
    """Compute temporal differences without crossing entity boundaries."""
    series = np.asarray(values, dtype=np.float64)
    entities = np.asarray(entity_ids)
    result = np.full(series.shape, np.nan, dtype=np.float64)
    if int(periods) >= series.size:
        return result
    current = np.arange(int(periods), series.size)
    previous = current - int(periods)
    same_entity = entities[current] == entities[previous]
    valid_current = current[same_entity]
    valid_previous = previous[same_entity]
    result[valid_current] = series[valid_current] - series[valid_previous]
    return result


@register_atom(witness_entity_time_aggregate)
@icontract.require(lambda values: _numeric_1d(values), "values must be a 1-D numeric array")
@icontract.require(lambda values, entity_ids: _same_1d_length(values, entity_ids), "values and entity_ids must align")
@icontract.require(lambda agg_fns: _valid_aggregations(agg_fns), "agg_fns must use supported aggregation names")
@icontract.ensure(lambda result, agg_fns: result.ndim == 2 and result.shape[1] == len(agg_fns), "aggregate output must have one column per aggregation")
@icontract.ensure(lambda result: _finite_or_nan(result), "aggregate output must be finite or NaN")
def entity_time_aggregate(
    values: NDArray[np.float64],
    entity_ids: NDArray[np.int64],
    agg_fns: list[str],
) -> NDArray[np.float64]:
    """Aggregate aligned time-series values into sorted entity-level summaries."""
    series = np.asarray(values, dtype=np.float64)
    entities = np.asarray(entity_ids)
    unique_entities = np.unique(entities)
    result = np.empty((unique_entities.size, len(agg_fns)), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for row, entity in enumerate(unique_entities):
            group_values = series[entities == entity]
            for col, agg in enumerate(agg_fns):
                if agg == "mean":
                    result[row, col] = float(np.nanmean(group_values))
                elif agg == "std":
                    result[row, col] = float(np.nanstd(group_values))
                elif agg == "min":
                    result[row, col] = float(np.nanmin(group_values))
                elif agg == "max":
                    result[row, col] = float(np.nanmax(group_values))
                else:
                    result[row, col] = float(np.count_nonzero(~np.isnan(group_values)))
    return result


@register_atom(witness_forward_fill)
@icontract.require(lambda series: np.asarray(series, dtype=np.float64).ndim in {1, 2}, "series must be 1-D or 2-D")
@icontract.require(lambda series: _finite_or_nan(series), "series must contain only finite values or NaN")
@icontract.require(lambda series, axis: -np.asarray(series).ndim <= int(axis) < np.asarray(series).ndim, "axis must be valid")
@icontract.ensure(lambda result, series: result.shape == np.asarray(series).shape, "forward fill must preserve shape")
@icontract.ensure(lambda result: _finite_or_nan(result), "forward-filled output must be finite or NaN")
def forward_fill(
    series: NDArray[np.float64],
    axis: int = 0,
) -> NDArray[np.float64]:
    """Propagate the last observed finite value forward along an axis."""
    values = np.asarray(series, dtype=np.float64)
    axis_index = int(axis) % values.ndim
    moved = np.moveaxis(values, axis_index, 0)
    valid = ~np.isnan(moved)
    index_shape = (moved.shape[0],) + (1,) * (moved.ndim - 1)
    base_indices = np.arange(moved.shape[0]).reshape(index_shape)
    last_valid = np.maximum.accumulate(np.where(valid, base_indices, 0), axis=0)
    filled = np.take_along_axis(moved, last_valid, axis=0)
    return np.moveaxis(filled, 0, axis_index)


@register_atom(witness_log1p_transform)
@icontract.require(lambda values: _numeric_1d(values, allow_nan=False), "values must be a finite 1-D numeric array")
@icontract.require(lambda values: np.all(np.asarray(values, dtype=np.float64) > -1.0), "log1p input must be greater than -1")
@icontract.ensure(lambda result, values: result.shape == np.asarray(values).shape, "log1p output must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "log1p output must be finite")
def log1p_transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the high-precision natural logarithm of one plus each value."""
    return np.log1p(np.asarray(values, dtype=np.float64)).astype(np.float64)


@register_atom(witness_expm1_inverse)
@icontract.require(lambda values: _numeric_1d(values, allow_nan=False), "values must be a finite 1-D numeric array")
@icontract.ensure(lambda result, values: result.shape == np.asarray(values).shape, "expm1 output must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "expm1 output must be finite")
def expm1_inverse(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert a log1p transform with the high-precision exponential minus one."""
    return np.expm1(np.asarray(values, dtype=np.float64)).astype(np.float64)


@register_atom(witness_exogenous_feature_concat)
@icontract.require(lambda sequence: _numeric_2d(sequence), "sequence must be a finite 2-D matrix")
@icontract.require(lambda exogenous_features: _numeric_1d(exogenous_features, allow_nan=False), "exogenous_features must be a finite 1-D vector")
@icontract.ensure(lambda result, sequence, exogenous_features: result.shape == (np.asarray(sequence).shape[0], np.asarray(sequence).shape[1] + np.asarray(exogenous_features).shape[0]), "concatenated output must append exogenous columns")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "concatenated output must be finite")
def exogenous_feature_concat(
    sequence: NDArray[np.float64],
    exogenous_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Broadcast static exogenous features across time and append them."""
    base = np.asarray(sequence, dtype=np.float64)
    static = np.asarray(exogenous_features, dtype=np.float64).reshape(1, -1)
    repeated = np.repeat(static, base.shape[0], axis=0)
    return np.concatenate([base, repeated], axis=1).astype(np.float64)
