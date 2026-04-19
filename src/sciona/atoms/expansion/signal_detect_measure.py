from __future__ import annotations

import numpy as np
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def witness_estimate_snr(
    signal: AbstractArray,
    noise_floor: AbstractScalar,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe SNR estimate and whether it clears the quality threshold."""
    return (
        AbstractScalar(dtype="float64"),
        AbstractScalar(dtype="bool"),
    )


def witness_analyze_peak_threshold_sensitivity(
    peaks: AbstractArray,
    threshold: AbstractScalar,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe threshold sensitivity and robustness flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
        AbstractScalar(dtype="bool"),
    )


def witness_check_event_rate_stationarity(
    event_times: AbstractArray,
    n_bins: AbstractScalar,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe stationarity coefficient-of-variation and pass/fail flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


def witness_estimate_false_positive_rate(
    detected_amplitudes: AbstractArray,
    noise_std: AbstractScalar,
    threshold: AbstractScalar,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe false-positive-rate estimate and reliability flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
        AbstractScalar(dtype="bool"),
    )


@register_atom(witness_estimate_snr)
def estimate_snr(
    signal: np.ndarray,
    noise_floor: float = 0.0,
) -> tuple[float, bool]:
    """Estimate signal-to-noise ratio in dB and flag whether it is acceptable."""
    s = np.asarray(signal, dtype=np.float64).ravel()
    if len(s) < 2:
        return 0.0, False

    signal_power = float(np.mean(s**2))
    if noise_floor <= 0:
        mad = float(np.median(np.abs(s - np.median(s))))
        noise_power = (mad * 1.4826) ** 2
    else:
        noise_power = noise_floor

    if noise_power == 0:
        return float("inf"), True

    snr = signal_power / noise_power
    snr_db = 10.0 * np.log10(max(snr, 1e-30))
    return float(snr_db), snr_db > 10.0


@register_atom(witness_analyze_peak_threshold_sensitivity)
def analyze_peak_threshold_sensitivity(
    peaks: np.ndarray,
    threshold: float,
) -> tuple[float, bool]:
    """Measure how many detected peaks sit near the decision threshold."""
    p = np.asarray(peaks, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if len(p) == 0 or not np.isfinite(threshold):
        return 0.0, True

    margin = max(abs(threshold) * 0.1, np.finfo(np.float64).eps)
    near_threshold = np.sum(np.abs(p - threshold) <= margin)
    sensitivity = float(near_threshold) / len(p)
    return sensitivity, sensitivity < 0.2


@register_atom(witness_check_event_rate_stationarity)
def check_event_rate_stationarity(
    event_times: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, bool]:
    """Estimate whether event arrivals remain roughly stationary over time."""
    t = np.asarray(event_times, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return 0.0, True

    t_min, t_max = float(np.min(t)), float(np.max(t))
    if t_max == t_min:
        return 0.0, True

    if not np.isfinite(n_bins):
        n_bins = 10
    n_bins = max(1, int(n_bins))
    bins = np.linspace(t_min, t_max, n_bins + 1)
    counts, _ = np.histogram(t, bins=bins)
    counts = counts.astype(float)
    mean_count = float(np.mean(counts))
    if mean_count == 0:
        return 0.0, True

    cv = float(np.std(counts)) / mean_count
    return cv, cv < 0.5


@register_atom(witness_estimate_false_positive_rate)
def estimate_false_positive_rate(
    detected_amplitudes: np.ndarray,
    noise_std: float,
    threshold: float,
) -> tuple[float, bool]:
    """Estimate the fraction of detections that are suspiciously close to noise."""
    d = np.asarray(detected_amplitudes, dtype=np.float64).ravel()
    if len(d) == 0 or noise_std <= 0:
        return 0.0, True

    suspect = np.sum(d < threshold + 2 * noise_std)
    fpr = float(suspect) / len(d)
    return fpr, fpr < 0.05


SIGNAL_DETECT_MEASURE_DECLARATIONS = {
    "estimate_snr": (
        "sciona.atoms.expansion.signal_detect_measure.estimate_snr",
        "ndarray, float -> tuple[float, bool]",
        "Estimate signal-to-noise ratio.",
    ),
    "analyze_peak_threshold_sensitivity": (
        "sciona.atoms.expansion.signal_detect_measure.analyze_peak_threshold_sensitivity",
        "ndarray, float -> tuple[float, bool]",
        "Analyze how sensitive detection count is to threshold changes.",
    ),
    "check_event_rate_stationarity": (
        "sciona.atoms.expansion.signal_detect_measure.check_event_rate_stationarity",
        "ndarray, int -> tuple[float, bool]",
        "Check whether the event rate is stationary over time.",
    ),
    "estimate_false_positive_rate": (
        "sciona.atoms.expansion.signal_detect_measure.estimate_false_positive_rate",
        "ndarray, float, float -> tuple[float, bool]",
        "Estimate the false positive detection rate.",
    ),
}
