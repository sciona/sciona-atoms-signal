import importlib


def test_e2e_ppg_kazemi_wrapper_d12_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12") is not None
    assert importlib.import_module("sciona.probes.signal_processing.e2e_ppg_kazemi_wrapper_d12") is not None
