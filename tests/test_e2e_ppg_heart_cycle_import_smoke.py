import importlib


def test_e2e_ppg_heart_cycle_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.heart_cycle") is not None
    assert importlib.import_module("sciona.probes.signal_processing.e2e_ppg_heart_cycle") is not None
