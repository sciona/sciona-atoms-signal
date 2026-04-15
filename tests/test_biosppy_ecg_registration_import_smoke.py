from __future__ import annotations

import importlib

from sciona.ghost.registry import list_registered


def test_biosppy_ecg_registration_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.ecg")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_ecg")
    assert hasattr(atoms, "bandpass_filter")
    assert hasattr(probes, "ECG_PROBE_TARGETS")
    registered = set(list_registered())
    assert "bandpass_filter" in registered
    assert "r_peak_detection" in registered
    assert "peak_correction" in registered
    assert "reject_outlier_intervals" in registered
    assert "template_extraction" in registered
    assert "heart_rate_computation" in registered
    assert "heart_rate_computation_median_smoothed" in registered
    assert "ssf_segmenter" in registered
    assert "christov_segmenter" in registered
