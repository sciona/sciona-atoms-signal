import importlib


def test_e2e_ppg_reconstruction_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.reconstruction") is not None
    assert importlib.import_module("sciona.probes.signal_processing.e2e_ppg_reconstruction") is not None
