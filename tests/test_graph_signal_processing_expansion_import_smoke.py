from __future__ import annotations

import importlib


def test_graph_signal_processing_expansion_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.expansion.graph_signal_processing")
    probes = importlib.import_module("sciona.probes.expansion.graph_signal_processing")
    assert hasattr(atoms, "validate_graph_connectivity")
    assert hasattr(probes, "GRAPH_SIGNAL_PROCESSING_PROBE_TARGETS")
