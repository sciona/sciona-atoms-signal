from __future__ import annotations

import importlib


def test_signal_filter_expansion_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.expansion.signal_filter")
    probes = importlib.import_module("sciona.probes.expansion.signal_filter")
    assert hasattr(atoms, "analyze_pole_stability")
    assert hasattr(probes, "SIGNAL_FILTER_PROBE_TARGETS")
