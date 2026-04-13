from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.signal_detect_measure"

SIGNAL_DETECT_MEASURE_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.estimate_snr", _MODULE, "estimate_snr"),
    ProbeTarget(
        f"{_MODULE}.analyze_peak_threshold_sensitivity",
        _MODULE,
        "analyze_peak_threshold_sensitivity",
    ),
    ProbeTarget(
        f"{_MODULE}.check_event_rate_stationarity",
        _MODULE,
        "check_event_rate_stationarity",
    ),
    ProbeTarget(
        f"{_MODULE}.estimate_false_positive_rate",
        _MODULE,
        "estimate_false_positive_rate",
    ),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SIGNAL_DETECT_MEASURE_PROBE_TARGETS
    ]
