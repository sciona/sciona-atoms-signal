import importlib


def test_neurokit2_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.neurokit2") is not None
    assert importlib.import_module("sciona.probes.signal_processing.neurokit2") is not None
