from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/licenses/provider_license.json"


def _load_manifest() -> dict:
    data = json.loads(MANIFEST_PATH.read_text())
    assert data["provider_repo"] == "sciona-atoms-signal"
    assert data["schema_version"] == "1.0"
    return data


def test_signal_provider_license_manifest_has_conservative_default_and_family_overrides() -> None:
    manifest = _load_manifest()

    default = manifest["repo_default"]
    assert default["scope"] == "repo"
    assert default["scope_key"] == "sciona-atoms-signal"
    assert default["license_expression"] == "NOASSERTION"
    assert default["license_status"] == "unknown"
    assert default["license_family"] == "unknown"
    assert default["source_kind"] == "manual_override"
    assert default["source_path"] is None

    overrides = manifest["family_overrides"]
    assert {entry["scope_key"] for entry in overrides} == {
        "sciona.atoms.signal_processing.biosppy",
        "sciona.atoms.signal_processing.neurokit2",
        "sciona.atoms.financial_signals",
        "sciona.atoms.anomaly_detection",
    }

    for entry in overrides:
        assert entry["scope"] == "family"
        assert entry["license_status"] == "approved"
        assert entry["license_family"] == "permissive"
        assert entry["source_kind"] in {"upstream_vendor_license", "manual_override"}
        assert entry["license_expression"] in {"BSD-3-Clause", "MIT"}
        assert entry["upstream_license_expression"] == entry["license_expression"]
        assert entry["source_path"].startswith(("package_metadata:", "external_vendor_tree:"))
        assert entry["notes"]
