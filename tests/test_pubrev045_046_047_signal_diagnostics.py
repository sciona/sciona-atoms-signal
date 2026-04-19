from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sciona.atoms.expansion.signal_detect_measure import (
    SIGNAL_DETECT_MEASURE_DECLARATIONS,
    analyze_peak_threshold_sensitivity,
    check_event_rate_stationarity,
)
from sciona.atoms.expansion.signal_filter import (
    analyze_group_delay_variation,
    measure_passband_ripple,
)
from sciona.atoms.expansion.signal_filter_registry import SIGNAL_FILTER_DECLARATIONS
from sciona.atoms.expansion.signal_transform import (
    analyze_window_leakage,
    detect_spectral_aliasing,
)
from sciona.atoms.expansion.signal_transform_registry import SIGNAL_TRANSFORM_DECLARATIONS
from sciona.probes.expansion.signal_detect_measure import SIGNAL_DETECT_MEASURE_PROBE_TARGETS
from sciona.probes.expansion.signal_filter import SIGNAL_FILTER_PROBE_TARGETS
from sciona.probes.expansion.signal_transform import SIGNAL_TRANSFORM_PROBE_TARGETS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = (
    ROOT
    / "data"
    / "review_bundles"
    / "signal_diagnostics_pubrev_045_046_047.review_bundle.json"
)
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "expansion" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

PUBREV_TARGETS = {
    "sciona.atoms.expansion.signal_detect_measure.analyze_peak_threshold_sensitivity",
    "sciona.atoms.expansion.signal_detect_measure.check_event_rate_stationarity",
    "sciona.atoms.expansion.signal_filter.analyze_group_delay_variation",
    "sciona.atoms.expansion.signal_filter.measure_passband_ripple",
    "sciona.atoms.expansion.signal_transform.analyze_window_leakage",
    "sciona.atoms.expansion.signal_transform.detect_spectral_aliasing",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pubrev045_046_047_review_bundle_is_catalog_ready() -> None:
    bundle = _json(BUNDLE_PATH)
    rows = {row["atom_key"]: row for row in bundle["rows"]}

    assert bundle["provider_repo"] == "sciona-atoms-signal"
    assert bundle["review_record_path"] == (
        "data/review_bundles/signal_diagnostics_pubrev_045_046_047.review_bundle.json"
    )
    assert set(rows) == PUBREV_TARGETS

    for atom_key, row in rows.items():
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["limitations"]
        assert row["review_record_path"] == bundle["review_record_path"]
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists(), f"{atom_key} source path is missing: {rel}"


def test_pubrev045_046_047_references_registry_declarations_and_probes_are_live() -> None:
    references = _json(REFERENCES_PATH)["atoms"]
    registry_ids = set(_json(REGISTRY_PATH)["references"])
    declaration_fqdns = {
        *(fqdn for fqdn, _, _ in SIGNAL_DETECT_MEASURE_DECLARATIONS.values()),
        *(fqdn for fqdn, _, _ in SIGNAL_FILTER_DECLARATIONS.values()),
        *(fqdn for fqdn, _, _ in SIGNAL_TRANSFORM_DECLARATIONS.values()),
    }
    probe_fqdns = {
        *(target.atom_fqdn for target in SIGNAL_DETECT_MEASURE_PROBE_TARGETS),
        *(target.atom_fqdn for target in SIGNAL_FILTER_PROBE_TARGETS),
        *(target.atom_fqdn for target in SIGNAL_TRANSFORM_PROBE_TARGETS),
    }

    assert PUBREV_TARGETS <= set(references)
    assert PUBREV_TARGETS <= declaration_fqdns
    assert PUBREV_TARGETS <= probe_fqdns

    for atom_key in PUBREV_TARGETS:
        entry = references[atom_key]
        assert entry["auto_attribution_runs"] == []
        assert entry["references"]
        assert {ref["ref_id"] for ref in entry["references"]} <= registry_ids
        for ref in entry["references"]:
            metadata = ref["match_metadata"]
            assert metadata["match_type"] == "manual"
            assert metadata["confidence"] in {"medium", "high"}


def test_pubrev045_signal_detect_measure_edge_cases_are_stable() -> None:
    sensitivity, stable = analyze_peak_threshold_sensitivity(
        np.array([9.9, 10.0, 10.1, np.nan, np.inf]),
        10.0,
    )
    assert sensitivity == 1.0
    assert stable is False

    sensitivity, stable = analyze_peak_threshold_sensitivity(np.array([0.0, 1.0]), 0.0)
    assert sensitivity == 0.5
    assert stable is False

    cv, stationary = check_event_rate_stationarity(
        np.array([0.0, 10.0, 20.0, 30.0, np.nan, np.inf]),
        n_bins=0,
    )
    assert cv == 0.0
    assert stationary is True


def test_pubrev046_signal_filter_diagnostics_handle_masks_and_nonfinite_values() -> None:
    ripple, ok = measure_passband_ripple(
        np.array([0.0, 0.5, np.inf, 1.5, 2.0]),
        np.array([True, True, True, False]),
    )
    assert ripple == 0.5
    assert ok is True

    variation, linear = analyze_group_delay_variation(np.array([3.0, np.nan, 3.4, np.inf]))
    assert np.isclose(variation, 0.4)
    assert linear is True


def test_pubrev047_window_leakage_uses_spectral_concentration_not_time_energy() -> None:
    n = np.arange(128, dtype=np.float64)
    tone = np.sin(2.0 * np.pi * 8.0 * n / len(n))
    hann_tone = tone * np.hanning(len(tone))

    tone_leakage, tone_excessive = analyze_window_leakage(tone, tone)
    hann_leakage, hann_excessive = analyze_window_leakage(hann_tone, tone)
    impulse_leakage, impulse_excessive = analyze_window_leakage(
        np.r_[1.0, np.zeros(127)],
        np.r_[1.0, np.zeros(127)],
    )

    assert tone_leakage < 1e-12
    assert tone_excessive is False
    assert hann_leakage < 0.01
    assert hann_excessive is False
    assert impulse_leakage > 0.9
    assert impulse_excessive is True


def test_pubrev047_spectral_aliasing_clamps_bad_thresholds() -> None:
    spectrum = np.zeros(100, dtype=np.complex128)
    spectrum[95] = 10.0
    fraction, aliased = detect_spectral_aliasing(spectrum)
    assert fraction == 1.0
    assert aliased is True

    fraction, aliased = detect_spectral_aliasing(spectrum, nyquist_fraction=1.5)
    assert fraction == 0.0
    assert aliased is False
