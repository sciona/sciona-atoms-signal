from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.signal_event_rate"

SIGNAL_EVENT_RATE_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.filter_signal_for_detection", _MODULE, "filter_signal_for_detection"),
    ProbeTarget(f"{_MODULE}.detect_peaks_in_signal", _MODULE, "detect_peaks_in_signal"),
    ProbeTarget(f"{_MODULE}.compute_event_rate", _MODULE, "compute_event_rate"),
    ProbeTarget(f"{_MODULE}.compute_event_rate_smoothed", _MODULE, "compute_event_rate_smoothed"),
    ProbeTarget(
        f"{_MODULE}.compute_event_rate_median_smoothed",
        _MODULE,
        "compute_event_rate_median_smoothed",
    ),
    ProbeTarget(f"{_MODULE}.assess_signal_quality", _MODULE, "assess_signal_quality"),
    ProbeTarget(f"{_MODULE}.remove_signal_jumps", _MODULE, "remove_signal_jumps"),
    ProbeTarget(f"{_MODULE}.reject_outlier_intervals", _MODULE, "reject_outlier_intervals"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SIGNAL_EVENT_RATE_PROBE_TARGETS
    ]
