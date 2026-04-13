from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_signal_event_rate_expansion_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.expansion.signal_event_rate")
    probes = importlib.import_module("sciona.probes.expansion.signal_event_rate")
    assert hasattr(atoms, "filter_signal_for_detection")
    assert hasattr(probes, "SIGNAL_EVENT_RATE_PROBE_TARGETS")


def test_signal_event_rate_expansion_asset_smoke() -> None:
    asset_path = Path(__file__).resolve().parents[1] / "data" / "expansions" / "signal_event_rate.json"
    asset = json.loads(asset_path.read_text())
    assert asset["family"] == "signal_event_rate"
    assert "signal_detect_measure" in asset.get("family_aliases", [])
