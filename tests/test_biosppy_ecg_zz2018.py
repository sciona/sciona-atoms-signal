import importlib


def test_biosppy_ecg_zz2018_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.ecg_zz2018")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_ecg_zz2018")
    assert hasattr(atoms, "calculatecompositesqi_zz2018")
    assert hasattr(atoms, "calculatebeatagreementsqi")
    assert hasattr(probes, "ECG_ZZ2018_PROBE_TARGETS")
