# REMEDIATION

This file tracks provider-owned signal atom debt that should not be papered over with relaxed publishability rules.

## Expansion

### `expansion.signal_event_rate` held pubrev-013 rows

Status: keep the listed rows unpublished for now.

Held atoms:
- `sciona.atoms.expansion.signal_event_rate.remove_signal_jumps`
- `sciona.atoms.expansion.signal_event_rate.reject_outlier_intervals`

Why they are blocked:
- `remove_signal_jumps` advertises step-discontinuity removal, but the current implementation returns the original waveform whenever first-difference MAD is near zero. A simple piecewise-constant waveform with one large step is therefore unchanged despite matching the public contract's main use case.
- `reject_outlier_intervals` advertises removal of implausible event intervals, but the current implementation only drops an interior event when both adjacent intervals are outside the MAD envelope. Common single-extra-event and zero-MAD interval cases are retained unchanged.
- Both helpers are conservative and may remain useful internally, but their current public names imply stronger correction semantics than the source establishes.

What we verified:
- The pubrev-013 safe provider subset is limited to `assess_signal_quality`, `detect_peaks_in_signal`, `compute_event_rate_smoothed`, `compute_event_rate_median_smoothed`, and `estimate_event_rate_from_signal`.
- The safe subset has provider-owned review bundle rows, references, limitations, probe metadata, and focused tests.
- `remove_signal_jumps` and `reject_outlier_intervals` remain absent from the provider review bundle's catalog-ready rows.

Proposed fixes:
1. For `remove_signal_jumps`, detect large absolute steps even when MAD is zero, or narrow the contract to noisy-signal jump correction only.
2. For `reject_outlier_intervals`, define whether the atom removes spurious extra events, missed-event gaps, both, or only double-sided local anomalies.
3. Add behavior tests for piecewise-constant steps, noisy steps, single extra events, missed-event gaps, and zero-MAD interval streams before reentering publication review.
