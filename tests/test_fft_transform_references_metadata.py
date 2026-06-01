"""References metadata tests for fft_transform."""

from __future__ import annotations

import json
from pathlib import Path


def test_fft_transform_references_cover_all_atoms() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    refs_path = repo_root / "src" / "sciona" / "atoms" / "signal" / "spectral" / "fft_transform" / "references.json"
    payload = json.loads(refs_path.read_text())

    atoms = payload["atoms"]
    assert set(atoms) == {
        "sciona.atoms.signal.spectral.fft_transform.atoms.apply_spectral_window@sciona/atoms/signal/spectral/fft_transform/atoms.py:22",
        "sciona.atoms.signal.spectral.fft_transform.atoms.optimize_fft_length@sciona/atoms/signal/spectral/fft_transform/atoms.py:49",
        "sciona.atoms.signal.spectral.fft_transform.atoms.compute_forward_rfft@sciona/atoms/signal/spectral/fft_transform/atoms.py:70",
    }

    # Verify that registry.json has these ref_ids
    registry_path = repo_root / "data" / "references" / "registry.json"
    registry = json.loads(registry_path.read_text())

    for atom_key in atoms:
        ref_ids = {ref["ref_id"] for ref in atoms[atom_key]["references"]}
        for ref_id in ref_ids:
            assert ref_id in registry["references"]
