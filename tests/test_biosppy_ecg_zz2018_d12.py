import importlib


def test_biosppy_ecg_zz2018_d12_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.ecg_zz2018_d12")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_ecg_zz2018_d12")
    assert hasattr(atoms, "assemble_zz2018_sqi")
    assert hasattr(atoms, "compute_beat_agreement_sqi")
    assert hasattr(probes, "ECG_ZZ2018_D12_PROBE_TARGETS")
