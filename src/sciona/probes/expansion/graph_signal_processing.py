from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.graph_signal_processing"

GRAPH_SIGNAL_PROCESSING_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(
        f"{_MODULE}.validate_graph_connectivity",
        _MODULE,
        "validate_graph_connectivity",
    ),
    ProbeTarget(
        f"{_MODULE}.check_laplacian_symmetry",
        _MODULE,
        "check_laplacian_symmetry",
    ),
    ProbeTarget(
        f"{_MODULE}.analyze_spectral_gap",
        _MODULE,
        "analyze_spectral_gap",
    ),
    ProbeTarget(
        f"{_MODULE}.validate_filter_response",
        _MODULE,
        "validate_filter_response",
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
        for target in GRAPH_SIGNAL_PROCESSING_PROBE_TARGETS
    ]
