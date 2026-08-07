from __future__ import annotations

import importlib


def test_neurokit2_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.neurokit2")
    probes = importlib.import_module("sciona.probes.signal_processing.neurokit2")

    assert hasattr(atoms, "zhao_2018_hrv_analysis")
    assert hasattr(atoms, "average_qrs_template")
    assert hasattr(probes, "NEUROKIT2_PROBE_TARGETS")
