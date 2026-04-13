from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.expansion.signal_transform"

SIGNAL_TRANSFORM_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.analyze_window_leakage", _MODULE, "analyze_window_leakage"),
    ProbeTarget(f"{_MODULE}.detect_spectral_aliasing", _MODULE, "detect_spectral_aliasing"),
    ProbeTarget(f"{_MODULE}.validate_parseval_energy", _MODULE, "validate_parseval_energy"),
    ProbeTarget(
        f"{_MODULE}.check_inverse_reconstruction",
        _MODULE,
        "check_inverse_reconstruction",
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
        for target in SIGNAL_TRANSFORM_PROBE_TARGETS
    ]
