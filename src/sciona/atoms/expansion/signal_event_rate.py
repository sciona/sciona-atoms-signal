from __future__ import annotations

import math

import icontract
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal
from sciona.ghost.registry import register_atom


def witness_filter_signal_for_detection(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Describe a conditioned waveform with the same envelope as the input signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain=signal.domain,
        units=signal.units,
    )


def witness_detect_peaks_in_signal(
    conditioned_signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractArray:
    """Describe detected peak locations as sorted integer indices."""
    return AbstractArray(
        shape=(conditioned_signal.shape[0],),
        dtype="int64",
        is_sorted=True,
        min_val=0,
        max_val=max(conditioned_signal.shape[0] - 1, 0),
    )


def witness_compute_event_rate(
    events: AbstractArray,
    sampling_rate: AbstractScalar,
) -> tuple[AbstractSignal, AbstractSignal]:
    """Describe midpoint indices and derived event-rate values."""
    length = max(events.shape[0] - 1, 0) if events.shape else 0
    midpoints = AbstractSignal(
        shape=(length,),
        dtype="int64",
        sampling_rate=1.0,
        domain="index",
        units="samples",
    )
    event_rate = AbstractSignal(
        shape=(length,),
        dtype="float64",
        sampling_rate=1.0,
        domain="measurement",
        units="events_per_minute",
    )
    return midpoints, event_rate


def witness_compute_event_rate_smoothed(
    events: AbstractArray,
    sampling_rate: AbstractScalar,
) -> tuple[AbstractSignal, AbstractSignal]:
    """Describe moving-average-smoothed event-rate values."""
    return witness_compute_event_rate(events, sampling_rate)


def witness_compute_event_rate_median_smoothed(
    events: AbstractArray,
    sampling_rate: AbstractScalar,
) -> tuple[AbstractSignal, AbstractSignal]:
    """Describe median-smoothed event-rate values."""
    return witness_compute_event_rate(events, sampling_rate)


def witness_estimate_event_rate_from_signal(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> tuple[AbstractArray, AbstractSignal, AbstractSignal]:
    """Describe detected events plus midpoint/rate outputs for a full signal-to-rate atom."""
    events = AbstractArray(
        shape=(signal.shape[0],),
        dtype="int64",
        is_sorted=True,
        min_val=0,
        max_val=max(signal.shape[0] - 1, 0),
    )
    midpoints, event_rate = witness_compute_event_rate_median_smoothed(events, sampling_rate)
    return events, midpoints, event_rate


def witness_assess_signal_quality(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> tuple[AbstractSignal, AbstractArray]:
    """Describe signal quality output as the original waveform plus a boolean mask."""
    quality_mask = AbstractArray(
        shape=signal.shape,
        dtype="bool",
    )
    return signal, quality_mask


def witness_remove_signal_jumps(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Describe a jump-corrected waveform with the same envelope as the input signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain=signal.domain,
        units=signal.units,
    )


def witness_reject_outlier_intervals(
    events: AbstractArray,
    sampling_rate: AbstractScalar,
) -> AbstractArray:
    """Describe a filtered, sorted event-index array."""
    return AbstractArray(
        shape=events.shape,
        dtype="int64",
        is_sorted=True,
        min_val=events.min_val,
        max_val=events.max_val,
    )


def _coerce_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        if not finite_mask.any():
            return np.zeros(0, dtype=np.float64)
        fill = float(np.median(values[finite_mask]))
        values = values.copy()
        values[~finite_mask] = fill
    return values


def _coerce_sampling_rate(sampling_rate: float | int) -> float:
    rate = float(sampling_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate!r}")
    return rate


def _valid_sampling_rate(sampling_rate: float | int) -> bool:
    try:
        rate = float(sampling_rate)
    except (TypeError, ValueError):
        return False
    return math.isfinite(rate) and rate > 0


def _robust_scale(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return mad
    std = float(np.std(values))
    return std if std > 0 else 1.0


@register_atom(witness_filter_signal_for_detection)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda filter_order: filter_order >= 1, "filter_order must be positive")
@icontract.require(lambda clipping_scale: clipping_scale > 0, "clipping_scale must be positive")
@icontract.require(lambda low_cutoff_hz, high_cutoff_hz: low_cutoff_hz > 0 and high_cutoff_hz > 0, "cutoffs must be positive")
@icontract.ensure(lambda result: result.ndim == 1 and np.isfinite(result).all(), "filtered signal must be finite and one-dimensional")
def filter_signal_for_detection(
    signal: np.ndarray,
    sampling_rate: float | int,
    *,
    filter_order: int = 4,
    clipping_scale: float = 8.0,
    low_cutoff_hz: float = 3.0,
    high_cutoff_hz: float = 25.0,
) -> np.ndarray:
    """Band-limit and clip a waveform before event detection."""
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(signal)
    if values.size == 0:
        return values

    centered = values - float(np.median(values))
    scale = _robust_scale(centered)
    clipped = np.clip(centered, -clipping_scale * scale, clipping_scale * scale)

    nyquist = rate / 2.0
    high = min(high_cutoff_hz, 0.45 * rate)
    low = min(low_cutoff_hz, high / 3.0)
    if low <= 0 or high <= low or high >= nyquist:
        return clipped

    sos = butter(
        filter_order,
        [low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, clipped)


def _pick_peak_orientation(
    values: np.ndarray,
    *,
    distance: int,
    prominence: float,
) -> np.ndarray:
    pos_peaks, pos_props = find_peaks(values, distance=distance, prominence=prominence)
    neg_peaks, neg_props = find_peaks(-values, distance=distance, prominence=prominence)
    pos_score = float(np.median(pos_props["prominences"])) if len(pos_peaks) else 0.0
    neg_score = float(np.median(neg_props["prominences"])) if len(neg_peaks) else 0.0
    return neg_peaks if neg_score > pos_score else pos_peaks


@register_atom(witness_detect_peaks_in_signal)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda prominence_scale: prominence_scale > 0, "prominence_scale must be positive")
@icontract.require(lambda refractory_scale: refractory_scale > 0, "refractory_scale must be positive")
@icontract.ensure(lambda result: result.dtype == np.int64 and np.all(np.diff(result) >= 0), "peaks must be sorted int64 indices")
def detect_peaks_in_signal(
    conditioned_signal: np.ndarray,
    sampling_rate: float | int,
    *,
    prominence_scale: float = 1.5,
    refractory_scale: float = 0.45,
) -> np.ndarray:
    """Detect robust event peaks in a conditioned waveform."""
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(conditioned_signal)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)

    scale = _robust_scale(values)
    prominence = max(prominence_scale * scale, 1e-6)
    distance = max(1, int(round(refractory_scale * rate)))
    peaks = _pick_peak_orientation(values, distance=distance, prominence=prominence)
    return np.asarray(peaks, dtype=np.int64)


@register_atom(witness_compute_event_rate)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.ensure(lambda result: len(result) == 2 and result[0].shape == result[1].shape, "midpoints and rates must align")
@icontract.ensure(lambda result: result[0].dtype == np.int64 and result[1].dtype == np.float64, "outputs must use stable dtypes")
@icontract.ensure(lambda result: np.isfinite(result[1]).all(), "event rates must be finite")
def compute_event_rate(
    events: np.ndarray,
    sampling_rate: float | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ordered event indices into midpoint locations and rate values."""
    rate = _coerce_sampling_rate(sampling_rate)
    event_idx = np.asarray(events, dtype=np.int64).reshape(-1)
    if event_idx.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    event_idx = np.unique(event_idx[event_idx >= 0])
    if event_idx.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    intervals = np.diff(event_idx).astype(np.float64)
    valid = intervals > 0
    if not np.any(valid):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    intervals = intervals[valid]
    left = event_idx[:-1][valid]
    midpoints = left + (intervals // 2).astype(np.int64)
    event_rate = 60.0 * rate / intervals
    return midpoints.astype(np.int64), event_rate.astype(np.float64)


@register_atom(witness_compute_event_rate_smoothed)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda smoothing_window: smoothing_window >= 1, "smoothing_window must be positive")
@icontract.ensure(lambda result: len(result) == 2 and result[0].shape == result[1].shape, "midpoints and rates must align")
@icontract.ensure(lambda result: np.isfinite(result[1]).all(), "smoothed rates must be finite")
def compute_event_rate_smoothed(
    events: np.ndarray,
    sampling_rate: float | int,
    *,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert event indices into a moving-average-smoothed rate estimate."""
    midpoints, event_rate = compute_event_rate(events, sampling_rate)
    if event_rate.size == 0:
        return midpoints, event_rate

    window = max(1, min(int(smoothing_window), int(event_rate.size)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(event_rate, kernel, mode="same")
    return midpoints, smoothed.astype(np.float64)


@register_atom(witness_compute_event_rate_median_smoothed)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda smoothing_window: smoothing_window >= 1, "smoothing_window must be positive")
@icontract.ensure(lambda result: len(result) == 2 and result[0].shape == result[1].shape, "midpoints and rates must align")
@icontract.ensure(lambda result: np.isfinite(result[1]).all(), "median-smoothed rates must be finite")
def compute_event_rate_median_smoothed(
    events: np.ndarray,
    sampling_rate: float | int,
    *,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert event indices into a median-smoothed rate estimate."""
    midpoints, event_rate = compute_event_rate(events, sampling_rate)
    if event_rate.size == 0:
        return midpoints, event_rate

    window = max(1, min(int(smoothing_window), int(event_rate.size)))
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return midpoints, event_rate

    smoothed = median_filter(event_rate.astype(np.float64), size=window, mode="nearest")
    return midpoints, np.asarray(smoothed, dtype=np.float64)


@register_atom(witness_estimate_event_rate_from_signal)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda smoothing_window: smoothing_window >= 1, "smoothing_window must be positive")
@icontract.ensure(lambda result: len(result) == 3, "result must contain events, midpoints, and rates")
@icontract.ensure(lambda result: result[1].shape == result[2].shape, "midpoints and rates must align")
@icontract.ensure(lambda result: np.isfinite(result[2]).all(), "estimated rates must be finite")
def estimate_event_rate_from_signal(
    signal: np.ndarray,
    sampling_rate: float | int,
    *,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate event locations and a robust event-rate series directly from a raw signal."""
    conditioned = filter_signal_for_detection(signal, sampling_rate)
    events = detect_peaks_in_signal(conditioned, sampling_rate)
    filtered_events = reject_outlier_intervals(events, sampling_rate)
    midpoints, event_rate = compute_event_rate_median_smoothed(
        filtered_events,
        sampling_rate,
        smoothing_window=smoothing_window,
    )
    return filtered_events, midpoints, event_rate


@register_atom(witness_assess_signal_quality)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda window_seconds: window_seconds > 0, "window_seconds must be positive")
@icontract.ensure(lambda result: len(result) == 2 and result[0].shape == result[1].shape, "signal and quality mask must align")
@icontract.ensure(lambda result: result[1].dtype == np.bool_, "quality mask must be boolean")
def assess_signal_quality(
    signal: np.ndarray,
    sampling_rate: float | int,
    *,
    window_seconds: float = 10.0,
    min_kurtosis: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark time windows whose local waveform statistics look unreliable."""
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(signal)
    if values.size == 0:
        return values, np.ones(0, dtype=bool)

    window = max(1, int(round(window_seconds * rate)))
    mask = np.ones(values.size, dtype=bool)
    for start in range(0, values.size, window):
        end = min(start + window, values.size)
        seg = values[start:end]
        if seg.size < 4:
            continue
        centered = seg - np.mean(seg)
        std = float(np.std(centered))
        if std < 1e-10:
            mask[start:end] = False
            continue
        kurt = float(np.mean((centered / std) ** 4)) - 3.0
        if kurt < min_kurtosis:
            mask[start:end] = False

    return values, mask


@register_atom(witness_remove_signal_jumps)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda jump_threshold_scale: jump_threshold_scale > 0, "jump_threshold_scale must be positive")
@icontract.ensure(lambda result: result.ndim == 1 and np.isfinite(result).all(), "corrected signal must be finite and one-dimensional")
def remove_signal_jumps(
    signal: np.ndarray,
    sampling_rate: float | int,
    *,
    jump_threshold_scale: float = 5.0,
) -> np.ndarray:
    """Remove large step discontinuities by flattening detected jumps."""
    _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(signal)
    if values.size < 2:
        return values

    diff = np.diff(values)
    median_diff = float(np.median(diff))
    mad_diff = float(np.median(np.abs(diff - median_diff)))
    threshold = 1e-10 if mad_diff < 1e-10 else jump_threshold_scale * mad_diff
    jumps = np.where(np.abs(diff - median_diff) > threshold)[0]
    if jumps.size == 0:
        return values

    result = values.copy()
    for idx in jumps:
        shift = result[idx + 1] - result[idx]
        result[idx + 1 :] -= shift
    return result


@register_atom(witness_reject_outlier_intervals)
@icontract.require(lambda sampling_rate: _valid_sampling_rate(sampling_rate), "sampling_rate must be positive")
@icontract.require(lambda events: np.asarray(events).size == 0 or bool(np.all(np.asarray(events) >= 0)), "events must be non-negative")
@icontract.require(lambda mad_scale: mad_scale > 0, "mad_scale must be positive")
@icontract.ensure(lambda result: result.dtype == np.int64 and np.all(np.diff(result) >= 0), "events must be sorted int64 indices")
def reject_outlier_intervals(
    events: np.ndarray,
    sampling_rate: float | int,
    *,
    mad_scale: float = 3.0,
) -> np.ndarray:
    """Drop event indices that participate in implausible local intervals."""
    _coerce_sampling_rate(sampling_rate)
    idx = np.asarray(events, dtype=np.int64).reshape(-1)
    if idx.size < 3:
        return idx

    idx = np.unique(idx)
    intervals = np.diff(idx).astype(np.float64)
    median_ivl = float(np.median(intervals))
    mad_ivl = float(np.median(np.abs(intervals - median_ivl)))
    tolerance = max(1.0, mad_scale * mad_ivl)
    lo = max(1.0, median_ivl - tolerance)
    hi = median_ivl + tolerance
    good = (intervals >= lo) & (intervals <= hi)
    if bool(good.all()):
        return idx

    keep = np.ones(idx.size, dtype=bool)
    candidates: list[tuple[float, int]] = []
    for i in range(1, idx.size - 1):
        if bool(good[i - 1]) and bool(good[i]):
            continue
        merged_interval = float(idx[i + 1] - idx[i - 1])
        merged_error = abs(merged_interval - median_ivl)
        if merged_error <= tolerance:
            candidates.append((merged_error, i))

    for _, i in sorted(candidates):
        if keep[i - 1] and keep[i + 1]:
            keep[i] = False
    return idx[keep]


SIGNAL_EVENT_RATE_DECLARATIONS = {
    "filter_signal_for_detection": (
        "sciona.atoms.expansion.signal_event_rate.filter_signal_for_detection",
        "np.ndarray, float -> np.ndarray",
        "Condition a sampled waveform for downstream peak/event detection.",
    ),
    "detect_peaks_in_signal": (
        "sciona.atoms.expansion.signal_event_rate.detect_peaks_in_signal",
        "np.ndarray, float -> np.ndarray",
        "Detect salient events in a conditioned waveform using robust thresholds.",
    ),
    "compute_event_rate": (
        "sciona.atoms.expansion.signal_event_rate.compute_event_rate",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Convert ordered event indices into midpoint indices and per-minute rate.",
    ),
    "compute_event_rate_smoothed": (
        "sciona.atoms.expansion.signal_event_rate.compute_event_rate_smoothed",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Convert ordered event indices into a smoothed per-minute rate estimate.",
    ),
    "compute_event_rate_median_smoothed": (
        "sciona.atoms.expansion.signal_event_rate.compute_event_rate_median_smoothed",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Convert ordered event indices into a robust median-smoothed rate estimate.",
    ),
    "estimate_event_rate_from_signal": (
        "sciona.atoms.expansion.signal_event_rate.estimate_event_rate_from_signal",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray, np.ndarray]",
        "Estimate event locations and a robust event-rate trace directly from a raw signal.",
    ),
    "assess_signal_quality": (
        "sciona.atoms.expansion.signal_event_rate.assess_signal_quality",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Compute per-window signal quality mask using kurtosis.",
    ),
    "remove_signal_jumps": (
        "sciona.atoms.expansion.signal_event_rate.remove_signal_jumps",
        "np.ndarray, float -> np.ndarray",
        "Remove step discontinuities from raw signal.",
    ),
    "reject_outlier_intervals": (
        "sciona.atoms.expansion.signal_event_rate.reject_outlier_intervals",
        "np.ndarray, float -> np.ndarray",
        "Remove events creating physiologically implausible intervals.",
    ),
}
