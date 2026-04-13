import importlib


def test_biosppy_svm_proc_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.svm_proc")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_svm_proc")
    assert hasattr(atoms, "get_auth_rates")
    assert hasattr(atoms, "majority_rule")
    assert hasattr(probes, "SVM_PROC_PROBE_TARGETS")
