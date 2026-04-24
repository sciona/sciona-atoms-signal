from __future__ import annotations

import json
from pathlib import Path


def test_financial_signals_references_cover_all_atoms() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    refs_path = repo_root / "src" / "sciona" / "atoms" / "financial_signals" / "references.json"
    payload = json.loads(refs_path.read_text())

    atoms = payload["atoms"]
    assert set(atoms) == {
        "sciona.atoms.financial_signals.atoms.realized_volatility@sciona/atoms/financial_signals/atoms.py:45",
        "sciona.atoms.financial_signals.atoms.realized_quadpower_quarticity@sciona/atoms/financial_signals/atoms.py:55",
        "sciona.atoms.financial_signals.atoms.weighted_average_price@sciona/atoms/financial_signals/atoms.py:79",
        "sciona.atoms.financial_signals.atoms.linear_trend_feature@sciona/atoms/financial_signals/atoms.py:101",
        "sciona.atoms.financial_signals.atoms.book_imbalance_features@sciona/atoms/financial_signals/atoms.py:115",
    }

    linear_refs = {
        ref["ref_id"]
        for ref in atoms["sciona.atoms.financial_signals.atoms.linear_trend_feature@sciona/atoms/financial_signals/atoms.py:101"]["references"]
    }
    assert linear_refs == {"repo_optiver_orvp", "scipy2020"}
