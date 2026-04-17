from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
from neurokit2.ecg.ecg_quality import _ecg_quality_averageQRS, _ecg_quality_zhao2018

from sciona.atoms.signal_processing.neurokit2 import averageqrstemplate, zhao2018hrvanalysis
from sciona.probes.signal_processing.neurokit2 import probe_records

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_ecg_signal(length: int, peak_indices: list[int]) -> np.ndarray:
    signal = np.zeros(length, dtype=float)
    kernel = np.array([0.05, 0.2, 0.7, 1.2, 0.7, 0.2, 0.05], dtype=float)
    half_width = len(kernel) // 2

    for peak in peak_indices:
        if half_width <= peak < length - half_width:
            signal[peak - half_width : peak + half_width + 1] += kernel

    signal += 0.02 * np.sin(np.linspace(0.0, 8.0 * np.pi, length))
    return signal


def test_neurokit2_probe_records_resolve_to_live_symbols() -> None:
    for record in probe_records():
        module = import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))
        fqdn_parts = str(record["atom_fqdn"]).split(".")
        imported = import_module(".".join(fqdn_parts[:-1]))
        assert getattr(imported, fqdn_parts[-1]) is getattr(module, str(record["wrapper_symbol"]))


def test_neurokit2_probe_records_publish_expected_symbols() -> None:
    assert {str(record["wrapper_symbol"]) for record in probe_records()} == {
        "averageqrstemplate",
        "zhao2018hrvanalysis",
    }


def test_neurokit2_references_map_to_live_atom_fqdns() -> None:
    references = json.loads(
        (ROOT / "src/sciona/atoms/signal_processing/neurokit2/references.json").read_text()
    )
    assert set(references["atoms"]) == {
        "sciona.atoms.signal_processing.neurokit2.averageqrstemplate@sciona/atoms/signal_processing/neurokit2/atoms.py:43",
        "sciona.atoms.signal_processing.neurokit2.zhao2018hrvanalysis@sciona/atoms/signal_processing/neurokit2/atoms.py:21",
    }


def test_averageqrstemplate_matches_upstream_quality_trace() -> None:
    signal = _synthetic_ecg_signal(
        5000,
        [400, 900, 1400, 1900, 2400, 2900, 3400, 3900, 4400],
    )
    rpeaks = np.array([400, 900, 1400, 1900, 2400, 2900, 3400, 3900, 4400], dtype=int)

    wrapped = averageqrstemplate(signal, rpeaks=rpeaks, sampling_rate=1000)
    upstream = _ecg_quality_averageQRS(signal, rpeaks=rpeaks, sampling_rate=1000)

    assert wrapped.shape == signal.shape
    assert np.allclose(wrapped, upstream)
    assert np.isfinite(wrapped).all()


def test_zhao2018hrvanalysis_matches_upstream_label() -> None:
    signal = _synthetic_ecg_signal(
        5000,
        [400, 900, 1400, 1900, 2400, 2900, 3400, 3900, 4400],
    )
    rpeaks = np.array([400, 900, 1400, 1900, 2400, 2900, 3400, 3900, 4400], dtype=int)

    wrapped = zhao2018hrvanalysis(
        signal,
        rpeaks=rpeaks,
        sampling_rate=1000,
        window=256,
        mode="simple",
    )
    upstream = _ecg_quality_zhao2018(
        signal,
        rpeaks=rpeaks,
        sampling_rate=1000,
        window=256,
        mode="simple",
    )

    assert wrapped == upstream
    assert wrapped in {"Excellent", "Barely acceptable", "Unacceptable"}
