"""Runtime atoms for Signal Filter expansion rules.

Provides deterministic, pure functions for filter design and
application diagnostics:

  - Pole-zero stability analysis (filter stability from pole locations)
  - Passband ripple measurement (frequency response quality)
  - Group delay variation analysis (phase distortion detection)
  - Transient response detection (startup artifact identification)
"""

from __future__ import annotations

import numpy as np
import icontract
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def witness_analyze_pole_stability(
    poles: AbstractArray,
    margin: AbstractScalar,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe pole magnitude diagnostics and stability flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


def witness_measure_passband_ripple(
    freq_response_db: AbstractArray,
    passband_mask: AbstractArray,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe passband ripple and acceptability flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


def witness_analyze_group_delay_variation(
    group_delay: AbstractArray,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe group-delay variation and linear-phase flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


def witness_detect_transient_response(
    output: AbstractArray,
    n_transient_samples: AbstractScalar,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe transient length and transient-energy fraction."""
    return (
        AbstractScalar(dtype="int64", min_val=0.0),
        AbstractScalar(dtype="float64", min_val=0.0, max_val=1.0),
    )


# ---------------------------------------------------------------------------
# Pole-zero stability analysis
# ---------------------------------------------------------------------------


@register_atom(witness_analyze_pole_stability)
@icontract.require(lambda poles: np.asarray(poles).size > 0, "poles must be non-empty")
@icontract.require(lambda margin: margin >= 0.0, "margin must be non-negative")
@icontract.ensure(lambda result: result[0] >= 0.0, "max pole magnitude must be non-negative")
def analyze_pole_stability(
    poles: np.ndarray,
    margin: float = 0.01,
) -> tuple[float, bool]:
    """Analyze filter stability from pole locations.

    For a causal stable filter, all poles must lie strictly inside
    the unit circle.  Poles near the unit circle indicate marginal
    stability.

    Args:
        poles: 1-D complex array of filter pole locations.
        margin: stability margin — poles with |z| > 1 - margin are risky.

    Returns:
        (max_pole_magnitude, is_stable) where is_stable is True if
        all poles have magnitude < 1.0.
    """
    p = np.asarray(poles, dtype=np.complex128).ravel()
    if len(p) == 0:
        return 0.0, True

    magnitudes = np.abs(p)
    max_mag = float(np.max(magnitudes))
    return max_mag, max_mag < 1.0


# ---------------------------------------------------------------------------
# Passband ripple measurement
# ---------------------------------------------------------------------------


@register_atom(witness_measure_passband_ripple)
@icontract.require(lambda freq_response_db: np.asarray(freq_response_db).size > 0, "freq_response_db must be non-empty")
@icontract.require(lambda passband_mask: np.asarray(passband_mask).size > 0, "passband_mask must be non-empty")
@icontract.require(lambda passband_mask: bool(np.any(np.asarray(passband_mask, dtype=bool))), "passband_mask must select samples")
@icontract.ensure(lambda result: result[0] >= 0.0, "ripple must be non-negative")
def measure_passband_ripple(
    freq_response_db: np.ndarray,
    passband_mask: np.ndarray,
) -> tuple[float, bool]:
    """Measure peak-to-peak ripple in the filter passband.

    Excessive passband ripple distorts signal amplitude.

    Args:
        freq_response_db: frequency response magnitude in dB.
        passband_mask: boolean mask selecting passband frequencies.

    Returns:
        (ripple_db, is_acceptable) where ripple_db is the
        peak-to-peak variation and is_acceptable is True if < 1.0 dB.
    """
    resp = np.asarray(freq_response_db, dtype=np.float64).ravel()
    mask = np.asarray(passband_mask, dtype=bool).ravel()

    if len(resp) == 0 or len(mask) == 0 or not np.any(mask):
        return 0.0, True

    n = min(len(resp), len(mask))
    passband = resp[:n][mask[:n]]
    passband = passband[np.isfinite(passband)]
    if len(passband) == 0:
        return 0.0, True

    ripple = float(np.max(passband) - np.min(passband))
    return ripple, ripple < 1.0


# ---------------------------------------------------------------------------
# Group delay variation
# ---------------------------------------------------------------------------


@register_atom(witness_analyze_group_delay_variation)
@icontract.require(lambda group_delay: np.asarray(group_delay).size >= 2, "group_delay must contain at least two samples")
@icontract.require(lambda group_delay: int(np.sum(np.isfinite(np.asarray(group_delay, dtype=np.float64)))) >= 2, "group_delay must contain at least two finite samples")
@icontract.ensure(lambda result: result[0] >= 0.0, "delay variation must be non-negative")
def analyze_group_delay_variation(
    group_delay: np.ndarray,
) -> tuple[float, bool]:
    """Analyze group delay variation across frequency.

    Large group delay variation indicates phase distortion that
    spreads signal energy across time.

    Args:
        group_delay: 1-D array of group delay values (samples) per frequency bin.

    Returns:
        (delay_variation, is_linear_phase) where delay_variation is
        max - min and is_linear_phase is True if variation < 1 sample.
    """
    gd = np.asarray(group_delay, dtype=np.float64).ravel()
    if len(gd) < 2:
        return 0.0, True

    finite = gd[np.isfinite(gd)]
    if len(finite) < 2:
        return 0.0, True

    variation = float(np.max(finite) - np.min(finite))
    return variation, variation < 1.0


# ---------------------------------------------------------------------------
# Transient response detection
# ---------------------------------------------------------------------------


@register_atom(witness_detect_transient_response)
@icontract.require(lambda output: np.asarray(output).size >= 4, "output must contain at least four samples")
@icontract.require(lambda output: bool(np.all(np.isfinite(np.asarray(output, dtype=np.float64)))), "output must be finite")
@icontract.require(lambda n_transient_samples: n_transient_samples >= 0, "n_transient_samples must be non-negative")
@icontract.ensure(lambda result: result[0] >= 0 and 0.0 <= result[1] <= 1.0, "transient length and energy fraction must be bounded")
def detect_transient_response(
    output: np.ndarray,
    n_transient_samples: int = 0,
) -> tuple[int, float]:
    """Detect startup transient in filter output.

    IIR filters produce a transient response before reaching steady
    state.  This function estimates the transient length as the number
    of samples before the running variance stabilizes.

    Args:
        output: 1-D filter output signal.
        n_transient_samples: expected transient length (0 = auto-detect).

    Returns:
        (estimated_transient_length, transient_energy_fraction) where
        transient_energy_fraction is the fraction of total energy in
        the transient region.
    """
    y = np.asarray(output, dtype=np.float64).ravel()
    n = len(y)
    if n < 4:
        return 0, 0.0

    if n_transient_samples > 0:
        t_len = min(n_transient_samples, n)
    else:
        # Auto-detect: find where running std stabilizes
        window = max(4, n // 20)
        stds = np.array([np.std(y[max(0, i - window):i + 1]) for i in range(n)])
        if len(stds) > window:
            median_std = float(np.median(stds[window:]))
            if median_std > 0:
                t_len = 0
                for i in range(n):
                    if abs(stds[i] - median_std) / median_std < 0.1:
                        t_len = i
                        break
                else:
                    t_len = n
            else:
                t_len = 0
        else:
            t_len = 0

    total_energy = float(np.sum(y ** 2))
    if total_energy == 0:
        return t_len, 0.0

    transient_energy = float(np.sum(y[:t_len] ** 2))
    return t_len, transient_energy / total_energy
