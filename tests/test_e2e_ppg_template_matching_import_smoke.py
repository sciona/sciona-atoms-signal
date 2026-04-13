import importlib


def test_e2e_ppg_template_matching_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.template_matching") is not None
    assert importlib.import_module("sciona.probes.signal_processing.e2e_ppg_template_matching") is not None
