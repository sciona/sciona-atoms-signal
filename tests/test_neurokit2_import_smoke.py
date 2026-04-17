from __future__ import annotations

import importlib


def test_neurokit2_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.neurokit2")
    probes = importlib.import_module("sciona.probes.signal_processing.neurokit2")

    assert hasattr(atoms, "zhao2018hrvanalysis")
    assert hasattr(atoms, "averageqrstemplate")
    assert hasattr(probes, "NEUROKIT2_PROBE_TARGETS")
