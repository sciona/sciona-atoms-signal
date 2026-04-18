from __future__ import annotations

import json
from pathlib import Path


def test_graph_signal_processing_references_cover_all_atoms() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    refs_path = repo_root / "src" / "sciona" / "atoms" / "expansion" / "references.json"
    payload = json.loads(refs_path.read_text())

    atoms = payload["atoms"]
    assert set(atoms) >= {
        "sciona.atoms.expansion.graph_signal_processing.validate_graph_connectivity",
        "sciona.atoms.expansion.graph_signal_processing.check_laplacian_symmetry",
        "sciona.atoms.expansion.graph_signal_processing.analyze_spectral_gap",
        "sciona.atoms.expansion.graph_signal_processing.validate_filter_response",
    }

    for atom_key in (
        "sciona.atoms.expansion.graph_signal_processing.validate_graph_connectivity",
        "sciona.atoms.expansion.graph_signal_processing.check_laplacian_symmetry",
        "sciona.atoms.expansion.graph_signal_processing.analyze_spectral_gap",
        "sciona.atoms.expansion.graph_signal_processing.validate_filter_response",
    ):
        ref_ids = {ref["ref_id"] for ref in atoms[atom_key]["references"]}
        assert ref_ids == {"shuman2013gsp", "scipy2020"}
