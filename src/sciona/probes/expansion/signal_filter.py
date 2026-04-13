from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.signal_filter"

SIGNAL_FILTER_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.analyze_pole_stability", _MODULE, "analyze_pole_stability"),
    ProbeTarget(f"{_MODULE}.measure_passband_ripple", _MODULE, "measure_passband_ripple"),
    ProbeTarget(
        f"{_MODULE}.analyze_group_delay_variation",
        _MODULE,
        "analyze_group_delay_variation",
    ),
    ProbeTarget(f"{_MODULE}.detect_transient_response", _MODULE, "detect_transient_response"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SIGNAL_FILTER_PROBE_TARGETS
    ]
