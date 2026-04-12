import importlib


def test_e2e_ppg_kazemi_wrapper_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper") is not None
    assert importlib.import_module("sciona.probes.signal_processing.e2e_ppg_kazemi_wrapper") is not None
