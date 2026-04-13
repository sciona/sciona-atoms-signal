from __future__ import annotations

import importlib


def test_signal_detect_measure_expansion_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.expansion.signal_detect_measure")
    probes = importlib.import_module("sciona.probes.expansion.signal_detect_measure")
    assert hasattr(atoms, "estimate_snr")
    assert hasattr(probes, "SIGNAL_DETECT_MEASURE_PROBE_TARGETS")
