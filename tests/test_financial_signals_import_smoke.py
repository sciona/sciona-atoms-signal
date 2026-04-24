from __future__ import annotations

import importlib

from sciona.ghost.registry import list_registered


def test_financial_signals_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.financial_signals")
    registered = set(list_registered())

    assert hasattr(atoms, "realized_volatility")
    assert hasattr(atoms, "realized_quadpower_quarticity")
    assert hasattr(atoms, "weighted_average_price")
    assert hasattr(atoms, "linear_trend_feature")
    assert hasattr(atoms, "book_imbalance_features")
    assert "realized_volatility" in registered
    assert "realized_quadpower_quarticity" in registered
    assert "weighted_average_price" in registered
    assert "linear_trend_feature" in registered
    assert "book_imbalance_features" in registered
