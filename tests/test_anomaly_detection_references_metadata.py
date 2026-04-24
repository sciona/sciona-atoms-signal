from __future__ import annotations

import json
from pathlib import Path


def test_anomaly_detection_references_cover_all_atoms() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    refs_path = repo_root / "src" / "sciona" / "atoms" / "anomaly_detection" / "references.json"
    payload = json.loads(refs_path.read_text())

    atoms = payload["atoms"]
    assert set(atoms) == {
        "sciona.atoms.anomaly_detection.atoms.matrix_profile_anomaly_score@sciona/atoms/anomaly_detection/atoms.py:145",
        "sciona.atoms.anomaly_detection.atoms.multiscale_anomaly_aggregation@sciona/atoms/anomaly_detection/atoms.py:161",
    }

    for atom_key in atoms:
        ref_ids = {ref["ref_id"] for ref in atoms[atom_key]["references"]}
        assert ref_ids == {"repo_kddcup2021_anomaly", "repo_stumpy"}
