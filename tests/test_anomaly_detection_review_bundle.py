"""Review bundle structure tests for anomaly_detection."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "review_bundles"


def _load_bundle(path: Path) -> dict:
    data = json.loads(path.read_text())
    assert data["review_record_path"] == f"data/review_bundles/{path.name}"
    return data


def test_anomaly_detection_review_bundle_structure() -> None:
    bundle = _load_bundle(BUNDLE_DIR / "anomaly_detection.review_bundle.json")

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


def test_anomaly_detection_review_bundle_covers_expected_rows() -> None:
    bundle = _load_bundle(BUNDLE_DIR / "anomaly_detection.review_bundle.json")
    assert {row["atom_key"] for row in bundle["rows"]} == {
        "sciona.atoms.anomaly_detection.matrix_profile_anomaly_score",
        "sciona.atoms.anomaly_detection.multiscale_anomaly_aggregation",
    }
