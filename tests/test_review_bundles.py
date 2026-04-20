from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "review_bundles"


def _load_bundle(path: Path) -> dict:
    data = json.loads(path.read_text())
    assert data["review_record_path"] == f"data/review_bundles/{path.name}"
    return data


def test_review_bundles_have_provider_owned_sources_and_rows() -> None:
    bundle_paths = sorted(BUNDLE_DIR.glob("*.json"))
    assert bundle_paths

    for path in bundle_paths:
        bundle = _load_bundle(path)

        assert bundle["provider_repo"] == "sciona-atoms-signal"
        assert bundle["review_status"] == "reviewed"
        assert bundle["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert bundle["review_developer_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert bundle["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}
        assert bundle["authoritative_sources"]
        assert bundle["rows"]

        for source in bundle["authoritative_sources"]:
            assert source["kind"]
            assert source.get("path")
            assert (ROOT / source["path"]).exists()

        for row in bundle["rows"]:
            assert row["atom_key"].startswith("sciona.atoms.")
            assert row["review_status"] in {"approved", "reviewed"}
            assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
            assert row["review_developer_semantic_verdict"] in {"pass", "pass_with_limits"}
            assert row["trust_readiness"] in {"catalog_ready", "needs_followup"}
            assert row["review_record_path"] == bundle["review_record_path"]
            assert row["source_paths"]
            for rel in row["source_paths"]:
                assert (ROOT / rel).exists()


def test_biosppy_online_filter_review_bundle_covers_expected_rows() -> None:
    bundle = _load_bundle(BUNDLE_DIR / "biosppy_online_filter.review_bundle.json")
    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.signal_processing.biosppy.online_filter.filterstateinit",
        "sciona.atoms.signal_processing.biosppy.online_filter.filterstep",
        "sciona.atoms.signal_processing.biosppy.online_filter_codex.filterstateinit",
        "sciona.atoms.signal_processing.biosppy.online_filter_codex.filterstep",
        "sciona.atoms.signal_processing.biosppy.online_filter_v2.filterstateinit",
        "sciona.atoms.signal_processing.biosppy.online_filter_v2.filterstep",
    }


def test_biosppy_svm_proc_review_bundle_covers_expected_rows() -> None:
    bundle = _load_bundle(BUNDLE_DIR / "biosppy_svm_proc.review_bundle.json")
    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.signal_processing.biosppy.svm_proc.assess_classification",
        "sciona.atoms.signal_processing.biosppy.svm_proc.assess_runs",
        "sciona.atoms.signal_processing.biosppy.svm_proc.combination",
        "sciona.atoms.signal_processing.biosppy.svm_proc.cross_validation",
        "sciona.atoms.signal_processing.biosppy.svm_proc.get_auth_rates",
        "sciona.atoms.signal_processing.biosppy.svm_proc.get_id_rates",
        "sciona.atoms.signal_processing.biosppy.svm_proc.get_subject_results",
        "sciona.atoms.signal_processing.biosppy.svm_proc.majority_rule",
    }


def test_neurokit2_review_bundle_covers_expected_rows() -> None:
    bundle = _load_bundle(BUNDLE_DIR / "neurokit2.review_bundle.json")
    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.signal_processing.neurokit2.averageqrstemplate",
        "sciona.atoms.signal_processing.neurokit2.zhao2018hrvanalysis",
    }


def test_e2e_ppg_review_bundle_covers_expected_rows() -> None:
    bundle = _load_bundle(BUNDLE_DIR / "e2e_ppg.review_bundle.json")
    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.signal_processing.e2e_ppg.heart_cycle.detect_heart_cycles",
        "sciona.atoms.signal_processing.e2e_ppg.heart_cycle.heart_cycle_detection",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.signalarraynormalization",
        "sciona.atoms.signal_processing.e2e_ppg.reconstruction.gan_patch_reconstruction",
        "sciona.atoms.signal_processing.e2e_ppg.reconstruction.windowed_signal_reconstruction",
        "sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction.generatereconstructedppg",
        "sciona.atoms.signal_processing.e2e_ppg.gan_reconstruction.gan_reconstruction",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12.normalizesignal",
        "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper_d12.wrapperevaluate",
        "sciona.atoms.signal_processing.e2e_ppg.template_matching.templatefeaturecomputation",
    }


def test_signal_expansion_review_bundle_covers_expected_rows() -> None:
    bundle = _load_bundle(BUNDLE_DIR / "signal_expansion.review_bundle.json")
    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.expansion.signal_event_rate.filter_signal_for_detection",
        "sciona.atoms.expansion.signal_event_rate.compute_event_rate",
        "sciona.atoms.expansion.signal_event_rate.assess_signal_quality",
        "sciona.atoms.expansion.signal_event_rate.detect_peaks_in_signal",
        "sciona.atoms.expansion.signal_event_rate.compute_event_rate_smoothed",
        "sciona.atoms.expansion.signal_event_rate.compute_event_rate_median_smoothed",
        "sciona.atoms.expansion.signal_event_rate.estimate_event_rate_from_signal",
        "sciona.atoms.expansion.signal_event_rate.remove_signal_jumps",
        "sciona.atoms.expansion.signal_event_rate.reject_outlier_intervals",
        "sciona.atoms.expansion.signal_filter.analyze_pole_stability",
        "sciona.atoms.expansion.signal_filter.detect_transient_response",
        "sciona.atoms.expansion.signal_transform.validate_parseval_energy",
        "sciona.atoms.expansion.signal_transform.check_inverse_reconstruction",
        "sciona.atoms.expansion.signal_detect_measure.estimate_snr",
        "sciona.atoms.expansion.signal_detect_measure.estimate_false_positive_rate",
        "sciona.atoms.expansion.graph_signal_processing.validate_graph_connectivity",
        "sciona.atoms.expansion.graph_signal_processing.check_laplacian_symmetry",
        "sciona.atoms.expansion.graph_signal_processing.analyze_spectral_gap",
        "sciona.atoms.expansion.graph_signal_processing.validate_filter_response",
    }
