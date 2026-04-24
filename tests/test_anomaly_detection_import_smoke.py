from __future__ import annotations

import importlib

from sciona.ghost.registry import list_registered


def test_anomaly_detection_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.anomaly_detection")
    registered = set(list_registered())

    assert hasattr(atoms, "matrix_profile_anomaly_score")
    assert hasattr(atoms, "multiscale_anomaly_aggregation")
    assert "matrix_profile_anomaly_score" in registered
    assert "multiscale_anomaly_aggregation" in registered
