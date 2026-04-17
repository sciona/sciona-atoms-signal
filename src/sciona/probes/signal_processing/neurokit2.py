"""Probe-side catalog for the NeuroKit2 ECG quality helper family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.neurokit2"

NEUROKIT2_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.zhao2018hrvanalysis", _MODULE, "zhao2018hrvanalysis"),
    ProbeTarget(f"{_MODULE}.averageqrstemplate", _MODULE, "averageqrstemplate"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in NEUROKIT2_PROBE_TARGETS
    ]
