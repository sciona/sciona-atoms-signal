import importlib


def test_heartpy_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.heartpy") is not None
    assert importlib.import_module("sciona.probes.signal_processing.heartpy") is not None
