from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PUBLISHABLE_ATOMS = {
    "sciona.atoms.signal_processing.e2e_ppg.heart_cycle.detect_heart_cycles",
    "sciona.atoms.signal_processing.e2e_ppg.heart_cycle.heart_cycle_detection",
    "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation",
    "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.signalarraynormalization",
}

HELD_ATOMS: set[str] = set()


def _load_json(relpath: str) -> dict:
    return json.loads((ROOT / relpath).read_text())


def _load_bundle() -> dict:
    return _load_json("data/review_bundles/e2e_ppg.review_bundle.json")


def test_pubrev_009_bundle_marks_safe_atoms_reference_ready() -> None:
    rows = {row["atom_key"]: row for row in _load_bundle()["rows"]}

    for atom_key in PUBLISHABLE_ATOMS:
        row = rows[atom_key]
        assert row["review_status"] == "approved"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["blocking_findings"] == []
        for relpath in row["source_paths"]:
            assert (ROOT / relpath).exists()

    for atom_key in HELD_ATOMS:
        row = rows[atom_key]
        assert row["trust_readiness"] == "needs_followup"
        assert "semantic_drift" in row["blocking_findings"]
        assert row["required_actions"]


def test_pubrev_009_references_cover_safe_atoms() -> None:
    heart_cycle_refs = _load_json("src/sciona/atoms/signal_processing/e2e_ppg/heart_cycle/references.json")
    kazemi_refs = _load_json("src/sciona/atoms/signal_processing/e2e_ppg/kazemi_wrapper/references.json")

    reference_keys = {
        key.split("@", 1)[0]
        for key in set(heart_cycle_refs["atoms"]) | set(kazemi_refs["atoms"])
    }
    for atom_key in PUBLISHABLE_ATOMS:
        assert atom_key in reference_keys

    for payload in heart_cycle_refs["atoms"].values():
        ref_ids = {ref["ref_id"] for ref in payload["references"]}
        assert {"kazemi2022ppg", "feli2023pipeline", "repo_e2e_ppg"} <= ref_ids


def test_pubrev_009_packages_export_selected_atoms() -> None:
    for atom_key in PUBLISHABLE_ATOMS | HELD_ATOMS:
        module_name, _, symbol_name = atom_key.rpartition(".")
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol_name)


def test_pubrev_009_heart_cycle_wrappers_delegate_to_upstream_module(monkeypatch) -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.heart_cycle.atoms")
    calls: list[tuple[np.ndarray, float]] = []

    def fake_detection(*, ppg: np.ndarray, sampling_rate: float) -> np.ndarray:
        calls.append((ppg, sampling_rate))
        return np.array([1, 3, 5])

    monkeypatch.setattr(
        atoms,
        "load_e2e_ppg_module",
        lambda module_name: SimpleNamespace(heart_cycle_detection=fake_detection),
    )

    ppg = np.array([0.0, 1.0, 0.2, 0.8])
    assert np.array_equal(atoms.detect_heart_cycles(ppg, 125.0), np.array([1, 3, 5]))
    assert np.array_equal(atoms.heart_cycle_detection(ppg, 125.0), np.array([1, 3, 5]))
    assert calls == [(ppg, 125.0), (ppg, 125.0)]


def test_pubrev_009_signalarraynormalization_loads_kazemi_module_lazily(monkeypatch) -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.atoms")
    loaded_modules: list[str] = []

    def fake_load(module_name: str) -> SimpleNamespace:
        loaded_modules.append(module_name)
        return SimpleNamespace(normalize=lambda arr: arr / np.max(np.abs(arr)))

    monkeypatch.setattr(atoms, "load_e2e_ppg_module", fake_load)

    arr = np.array([0.0, 2.0, 4.0])
    assert np.allclose(atoms.signalarraynormalization(arr), np.array([0.0, 0.5, 1.0]))
    assert loaded_modules == ["kazemi_peak_detection"]
