"""Matrix-profile anomaly atoms adapted from the KDD 2021 anomaly solution."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray
import stumpy

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_matrix_profile_anomaly_score,
    witness_multiscale_anomaly_aggregation,
)


_DENOM_THRESHOLD = 0.1
_UPPER_THRESHOLD = 0.75
_LOWER_THRESHOLD = 0.25
_CONST_THRESHOLD = 0.05
_MIN_COEF = 0.5
_PADDING_LENGTH = 3


def _is_finite_array(values: NDArray[np.float64]) -> bool:
    """Return whether an ndarray contains only finite values."""
    return bool(np.isfinite(values).all())


def _is_valid_window_sizes(series: NDArray[np.float64], window_sizes: Sequence[int]) -> bool:
    """Return whether every window size is a usable integer subsequence length."""
    return (
        len(window_sizes) >= 1
        and all(isinstance(window, int) for window in window_sizes)
        and all(3 <= window < series.shape[0] for window in window_sizes)
    )


def _moving_average(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Compute a simple same-length moving average."""
    if values.size == 0:
        return values.copy()
    width = max(1, min(int(width), int(values.size)))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values, kernel, mode="same")


def _peak_to_peak_windows(series: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
    """Compute peak-to-peak range for every subsequence window."""
    windows = np.lib.stride_tricks.sliding_window_view(series, window_size)
    return windows.max(axis=1) - windows.min(axis=1)


def _variation_coef(orig_p2p: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
    """Compute the low-variation penalty term adapted from the KDD solution."""
    if orig_p2p.size == 0:
        return orig_p2p.copy()

    mean = float(np.mean(orig_p2p))
    if mean <= 0.0:
        return np.zeros_like(orig_p2p, dtype=np.float64)

    upper = mean * _UPPER_THRESHOLD
    lower = mean * _LOWER_THRESHOLD
    const = mean * _CONST_THRESHOLD
    scale = upper - lower

    if scale <= 0.0:
        coef = np.ones_like(orig_p2p, dtype=np.float64)
    else:
        coef = np.clip((orig_p2p - lower) / scale, 0.0, 1.0)

    low_change = orig_p2p <= const
    if np.any(low_change):
        dilation_width = max(1, min(2 * window_size, low_change.size))
        dilated = np.convolve(
            low_change.astype(np.float64),
            np.ones(dilation_width, dtype=np.float64),
            mode="same",
        ) > 0.0
        coef[dilated] = 0.0

    if float(np.mean(coef)) < _MIN_COEF:
        coef.fill(0.0)

    return coef


def _normalize_unit_interval(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a non-negative score vector to [0, 1]."""
    if values.size == 0:
        return values.copy()
    max_val = float(np.max(values))
    if max_val <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / max_val


def _matrix_profile_score_core(
    series: NDArray[np.float64],
    window_size: int,
) -> NDArray[np.float64]:
    """Compute the normalized self-join anomaly score per subsequence."""
    matrix_profile = stumpy.stump(series, window_size)
    mp_values = matrix_profile[:, 0].astype(np.float64)
    mp_indices = matrix_profile[:, 1].astype(np.int64)

    orig_p2p = _peak_to_peak_windows(series, window_size)
    coef = _variation_coef(orig_p2p, window_size)
    weighted_profile = mp_values * coef

    reference = mp_values[mp_indices]
    raw = np.zeros_like(weighted_profile, dtype=np.float64)
    positive_reference = reference > 0.0
    np.divide(weighted_profile, reference, out=raw, where=positive_reference)

    saturated = (~positive_reference) & (weighted_profile > 0.0)
    raw[saturated] = 1.0 / _DENOM_THRESHOLD
    raw = np.clip(raw, 0.0, 1.0 / _DENOM_THRESHOLD)

    smoothed = _moving_average(raw, window_size)

    padding = window_size * _PADDING_LENGTH
    mask = np.zeros_like(smoothed, dtype=np.float64)
    left = min(window_size, mask.size)
    right = max(left, mask.size - window_size - padding)
    mask[left:right] = 1.0
    if padding > 0:
        mask = _moving_average(mask, padding)

    return _normalize_unit_interval(smoothed * mask)


@register_atom(witness_matrix_profile_anomaly_score)
@icontract.require(lambda series: series.ndim == 1, "series must be 1-D")
@icontract.require(lambda series: _is_finite_array(series), "series must be finite")
@icontract.require(lambda series, window_size: 3 <= window_size < series.shape[0], "window_size must be >= 3 and smaller than the series length")
@icontract.ensure(lambda result, series, window_size: result.shape == (series.shape[0] - window_size + 1,), "result must contain one score per subsequence")
@icontract.ensure(lambda result: _is_finite_array(result), "result must be finite")
@icontract.ensure(lambda result: np.all(result >= 0.0), "result must be non-negative")
@icontract.ensure(lambda result: np.all(result <= 1.0 + 1e-12), "result must be normalized to [0, 1]")
def matrix_profile_anomaly_score(
    series: NDArray[np.float64],
    window_size: int,
) -> NDArray[np.float64]:
    """Compute a normalized self-join matrix-profile anomaly score."""
    return _matrix_profile_score_core(series, window_size)


@register_atom(witness_multiscale_anomaly_aggregation)
@icontract.require(lambda series: series.ndim == 1, "series must be 1-D")
@icontract.require(lambda series: _is_finite_array(series), "series must be finite")
@icontract.require(_is_valid_window_sizes, "window_sizes must be a non-empty list of valid integers")
@icontract.ensure(lambda result, series: result.shape == series.shape, "result must align to the original series")
@icontract.ensure(lambda result: _is_finite_array(result), "result must be finite")
@icontract.ensure(lambda result: np.all(result >= 0.0), "result must be non-negative")
@icontract.ensure(lambda result: np.all(result <= 1.0 + 1e-12), "result must be normalized to [0, 1]")
def multiscale_anomaly_aggregation(
    series: NDArray[np.float64],
    window_sizes: list[int],
) -> NDArray[np.float64]:
    """Aggregate multi-scale matrix-profile scores by max-pooling over support.

    Each scale contributes a normalized subsequence score. The score is then
    lifted back onto the original time axis by max-pooling over the covered
    samples, mirroring the KDD solution's window-wise peak search across scales.
    """
    aggregate = np.zeros(series.shape[0], dtype=np.float64)

    for window_size in sorted(set(window_sizes)):
        scale_score = _matrix_profile_score_core(series, window_size)
        lifted = np.zeros_like(aggregate)
        for start, value in enumerate(scale_score):
            if value <= 0.0:
                continue
            lifted[start : start + window_size] = np.maximum(
                lifted[start : start + window_size],
                value,
            )
        aggregate = np.maximum(aggregate, lifted)

    return _normalize_unit_interval(aggregate)
