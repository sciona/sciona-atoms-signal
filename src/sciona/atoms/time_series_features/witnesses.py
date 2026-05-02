"""Ghost witnesses for time-series feature atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_rolling_window_features(
    series: AbstractArray,
    window_size: int,
    statistics: list[str],
) -> AbstractArray:
    """Describe a same-length rolling feature matrix."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(series.shape) != 1:
        raise ValueError("series must be 1D")
    return AbstractArray(shape=(series.shape[0], len(statistics)), dtype="float64")


def witness_create_lag_features(
    values: AbstractArray,
    lags: list[int],
) -> AbstractArray:
    """Describe a same-length lag feature matrix."""
    if len(values.shape) != 1:
        raise ValueError("values must be 1D")
    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must be positive")
    return AbstractArray(shape=(values.shape[0], len(lags)), dtype="float64")


def witness_technical_indicators_macd(
    prices: AbstractArray,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe three aligned MACD output series."""
    if not slow > fast > 0 or signal <= 0:
        raise ValueError("invalid MACD spans")
    return (
        AbstractArray(shape=prices.shape, dtype="float64"),
        AbstractArray(shape=prices.shape, dtype="float64"),
        AbstractArray(shape=prices.shape, dtype="float64"),
    )


def witness_seasonal_decompose_additive(
    series: AbstractArray,
    period: int,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe aligned additive decomposition outputs."""
    if period <= 1:
        raise ValueError("period must exceed one")
    return (
        AbstractArray(shape=series.shape, dtype="float64"),
        AbstractArray(shape=series.shape, dtype="float64"),
        AbstractArray(shape=series.shape, dtype="float64"),
    )


def witness_grouped_temporal_diff(
    values: AbstractArray,
    entity_ids: AbstractArray,
    periods: int,
) -> AbstractArray:
    """Describe a same-length grouped temporal difference vector."""
    if periods <= 0:
        raise ValueError("periods must be positive")
    if values.shape != entity_ids.shape:
        raise ValueError("values and entity_ids must align")
    return AbstractArray(shape=values.shape, dtype="float64")


def witness_entity_time_aggregate(
    values: AbstractArray,
    entity_ids: AbstractArray,
    agg_fns: list[str],
) -> AbstractArray:
    """Describe entity rows by requested aggregation columns."""
    if values.shape != entity_ids.shape:
        raise ValueError("values and entity_ids must align")
    return AbstractArray(shape=(values.shape[0], len(agg_fns)), dtype="float64")


def witness_forward_fill(
    series: AbstractArray,
    axis: int = 0,
) -> AbstractArray:
    """Describe shape-preserving forward fill."""
    if not -len(series.shape) <= axis < len(series.shape):
        raise ValueError("axis out of range")
    return AbstractArray(shape=series.shape, dtype=series.dtype)


def witness_log1p_transform(values: AbstractArray) -> AbstractArray:
    """Describe shape-preserving log1p transform."""
    return AbstractArray(shape=values.shape, dtype="float64")


def witness_expm1_inverse(values: AbstractArray) -> AbstractArray:
    """Describe shape-preserving expm1 inverse transform."""
    return AbstractArray(shape=values.shape, dtype="float64")


def witness_exogenous_feature_concat(
    sequence: AbstractArray,
    exogenous_features: AbstractArray,
) -> AbstractArray:
    """Describe time-axis broadcast and feature concatenation."""
    if len(sequence.shape) != 2 or len(exogenous_features.shape) != 1:
        raise ValueError("sequence must be 2D and exogenous_features must be 1D")
    return AbstractArray(
        shape=(sequence.shape[0], sequence.shape[1] + exogenous_features.shape[0]),
        dtype="float64",
    )


def witness_interpolate_to_timestamps(
    source_times: AbstractArray,
    source_values: AbstractArray,
    target_times: AbstractArray,
) -> AbstractArray:
    """Describe linearly interpolated values at target timestamps."""
    if source_times.shape != source_values.shape:
        raise ValueError("source_times and source_values must have equal shape")
    if source_times.shape[0] < 2:
        raise ValueError("source_times must contain at least two samples")
    return AbstractArray(shape=target_times.shape, dtype="float64")
