from __future__ import annotations

import importlib


def test_signal_transform_expansion_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.expansion.signal_transform")
    probes = importlib.import_module("sciona.probes.expansion.signal_transform")
    assert hasattr(atoms, "analyze_window_leakage")
    assert hasattr(probes, "SIGNAL_TRANSFORM_PROBE_TARGETS")
