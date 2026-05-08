"""Probe-side catalog for the BioSPPy ECG ZZ2018 D12 signal-quality family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.ecg_zz2018_d12"

ECG_ZZ2018_D12_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.assemble_zz2018_sqi", _MODULE, "assemble_zz2018_sqi"),
    ProbeTarget(f"{_MODULE}.compute_beat_agreement_sqi", _MODULE, "compute_beat_agreement_sqi"),
    ProbeTarget(f"{_MODULE}.compute_frequency_sqi", _MODULE, "compute_frequency_sqi"),
    ProbeTarget(f"{_MODULE}.compute_kurtosis_sqi", _MODULE, "compute_kurtosis_sqi"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ECG_ZZ2018_D12_PROBE_TARGETS
    ]
