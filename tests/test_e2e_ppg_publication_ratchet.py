from __future__ import annotations

import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_bundle() -> dict:
    return json.loads((ROOT / "data" / "review_bundles" / "e2e_ppg.review_bundle.json").read_text())


def _load_json(relpath: str) -> object:
    return json.loads((ROOT / relpath).read_text())


def test_e2e_ppg_publication_bundle_rows() -> None:
    bundle = _load_bundle()
    for source in bundle["authoritative_sources"]:
        assert (ROOT / source["path"]).exists()
    for row in bundle["rows"]:
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists()

    gan_refs = _load_json("src/sciona/atoms/signal_processing/e2e_ppg/gan_reconstruction/references.json")
    template_refs = _load_json("src/sciona/atoms/signal_processing/e2e_ppg/template_matching/references.json")
    d12_refs = _load_json("src/sciona/atoms/signal_processing/e2e_ppg/kazemi_wrapper_d12/references.json")
    _load_json("src/sciona/atoms/signal_processing/e2e_ppg/gan_reconstruction/cdg.json")
    _load_json("src/sciona/atoms/signal_processing/e2e_ppg/gan_reconstruction/matches.json")
    _load_json("src/sciona/atoms/signal_processing/e2e_ppg/template_matching/cdg.json")
    _load_json("src/sciona/atoms/signal_processing/e2e_ppg/template_matching/matches.json")

    assert set(gan_refs["atoms"]) == {
        "sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction.generate_reconstructed_ppg@sciona/atoms/signal_processing/e2e_ppg/gan_reconstruction/atoms.py:22",
        "sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction.gan_reconstruction@sciona/atoms/signal_processing/e2e_ppg/gan_reconstruction/atoms.py:47",
    }
    assert set(template_refs["atoms"]) == {
        "sciona.atoms.signal_processing.e2e_ppg.template_matching.template_feature_computation@sciona/atoms/signal_processing/e2e_ppg/template_matching/atoms.py:16",
    }
    assert set(d12_refs["atoms"]) == {
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12.normalize_signal@sciona/atoms/signal_processing/e2e_ppg/kazemi_wrapper_d12/atoms.py:25",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12.wrapper_evaluate@sciona/atoms/signal_processing/e2e_ppg/kazemi_wrapper_d12/atoms.py:34",
    }
    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.signal_processing.e2e_ppg.heart_cycle.detect_heart_cycles",
        "sciona.atoms.signal_processing.e2e_ppg.heart_cycle.heart_cycle_detection",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.wrapper_prediction_signal_computation",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.signal_array_normalization",
        "sciona.atoms.signal_processing.e2e_ppg.reconstruction.gan_patch_reconstruction",
        "sciona.atoms.signal_processing.e2e_ppg.reconstruction.windowed_signal_reconstruction",
        "sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction.generate_reconstructed_ppg",
        "sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction.gan_reconstruction",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12.normalize_signal",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12.wrapper_evaluate",
        "sciona.atoms.signal_processing.e2e_ppg.template_matching.template_feature_computation",
    }


def test_e2e_ppg_publication_packages_export_atoms() -> None:
    gan_pkg = importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction")
    gan_atoms = importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction.atoms")
    template_pkg = importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.template_matching")
    template_atoms = importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.template_matching.atoms")
    d12_pkg = importlib.import_module("sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12")

    assert hasattr(gan_pkg, "generate_reconstructed_ppg")
    assert hasattr(gan_pkg, "gan_reconstruction")
    assert hasattr(gan_atoms, "generate_reconstructed_ppg")
    assert hasattr(gan_atoms, "gan_reconstruction")
    assert hasattr(template_pkg, "template_feature_computation")
    assert hasattr(template_atoms, "template_feature_computation")
    assert hasattr(d12_pkg, "normalize_signal")
    assert hasattr(d12_pkg, "wrapper_evaluate")
