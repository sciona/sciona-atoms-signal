from __future__ import annotations

import json
from pathlib import Path


def test_signal_references_use_canonical_atoms_module_segments() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    signal_root = src_root / "sciona" / "atoms" / "signal_processing"

    for refs_path in sorted(signal_root.rglob("references.json")):
        payload = json.loads(refs_path.read_text())
        atoms = payload.get("atoms", {})
        if not isinstance(atoms, dict) or not atoms:
            continue

        module_prefix = ".".join(refs_path.parent.relative_to(src_root).parts)
        canonical_prefix = f"{module_prefix}.atoms."

        for atom_key in atoms:
            assert atom_key.startswith(canonical_prefix), (
                f"{refs_path} still uses a non-canonical key: {atom_key!r}. "
                f"Expected prefix {canonical_prefix!r}"
            )
