from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper import wrapperpredictionsignalcomputation


ROOT = Path(__file__).resolve().parents[1]
ATOM = "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation"
REFERENCE_ATOM = "sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.atoms.wrapperpredictionsignalcomputation"


def test_wrapper_prediction_signal_computation_extracts_source_aligned_peak_indices() -> None:
    prediction = np.zeros(80, dtype=float)
    prediction[[5, 10, 50]] = 1.0
    prediction[20] = 0.95

    raw_signal = np.zeros(80, dtype=float)
    raw_signal[5] = 1.0
    raw_signal[10] = 2.0
    raw_signal[20] = 1.0
    raw_signal[50] = 3.0

    result = wrapperpredictionsignalcomputation(prediction, raw_signal)

    assert result.dtype.kind in {"i", "u"}
    assert np.array_equal(result, np.array([10, 50]))


def test_wrapper_prediction_signal_computation_filters_nonpositive_candidates() -> None:
    prediction = np.zeros(24, dtype=float)
    prediction[5] = 1.0
    raw_signal = np.ones(24, dtype=float)
    raw_signal[5] = 0.0

    assert np.array_equal(wrapperpredictionsignalcomputation(prediction, raw_signal), np.array([], dtype=np.intp))


def test_wrapper_prediction_signal_computation_preserves_upstream_tail_tie_rule() -> None:
    prediction = np.zeros(24, dtype=float)
    prediction[18] = 1.0
    raw_signal = np.ones(24, dtype=float)

    assert np.array_equal(wrapperpredictionsignalcomputation(prediction, raw_signal), np.array([], dtype=np.intp))

    prediction[20] = 1.0
    raw_signal[18] = 1.0
    raw_signal[20] = 2.0
    assert np.array_equal(wrapperpredictionsignalcomputation(prediction, raw_signal), np.array([20]))


def test_wrapper_prediction_signal_computation_accepts_column_vectors_and_constant_scores() -> None:
    prediction = np.ones((24, 1), dtype=float)
    raw_signal = np.ones((24, 1), dtype=float)

    assert np.array_equal(wrapperpredictionsignalcomputation(prediction, raw_signal), np.array([], dtype=np.intp))


def test_kazemi_wrapper_reingest_metadata_marks_atom_publishable() -> None:
    bundle = json.loads((ROOT / "data/review_bundles/e2e_ppg.review_bundle.json").read_text())
    row = {item["atom_key"]: item for item in bundle["rows"]}[ATOM]

    assert row["review_status"] == "approved"
    assert row["trust_readiness"] == "catalog_ready"
    assert row["blocking_findings"] == []
    assert row["required_actions"] == []
    assert row["has_references"] is True
    assert row["references_status"] == "pass"

    refs = json.loads(
        (
            ROOT
            / "src/sciona/atoms/signal_processing/e2e_ppg/kazemi_wrapper/references.json"
        ).read_text()
    )
    reference_key = next(key for key in refs["atoms"] if key.startswith(f"{REFERENCE_ATOM}@"))
    ref_ids = {ref["ref_id"] for ref in refs["atoms"][reference_key]["references"]}
    assert {"kazemi2022ppg", "feli2023pipeline", "repo_e2e_ppg"} <= ref_ids

    manifest = json.loads((ROOT / "data/audit_manifest.json").read_text())
    manifest_row = {item["atom_key"]: item for item in manifest["atoms"]}[ATOM]
    assert manifest_row["review_status"] == "approved"
    assert manifest_row["trust_readiness"] == "reviewed_with_limits"
    assert manifest_row["blocking_findings"] == []
    assert manifest_row["required_actions"] == []
    assert manifest_row["parity_coverage_level"] == "positive_and_negative"
