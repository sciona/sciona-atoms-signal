import importlib


def test_e2e_ppg_gan_reconstruction_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction") is not None
    assert importlib.import_module("sciona.probes.signal_processing.e2e_ppg_gan_reconstruction") is not None
