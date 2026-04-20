from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from sciona.atoms.expansion.signal_event_rate import (
    SIGNAL_EVENT_RATE_DECLARATIONS,
    assess_signal_quality,
    compute_event_rate,
    compute_event_rate_median_smoothed,
    compute_event_rate_smoothed,
    detect_peaks_in_signal,
    estimate_event_rate_from_signal,
    remove_signal_jumps,
    reject_outlier_intervals,
)
from sciona.probes.expansion.signal_event_rate import SIGNAL_EVENT_RATE_PROBE_TARGETS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "signal_expansion.review_bundle.json"
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "expansion" / "references.json"
SAFE_PUBREV_013_KEYS = {
    "sciona.atoms.expansion.signal_event_rate.assess_signal_quality",
    "sciona.atoms.expansion.signal_event_rate.detect_peaks_in_signal",
    "sciona.atoms.expansion.signal_event_rate.compute_event_rate_smoothed",
    "sciona.atoms.expansion.signal_event_rate.compute_event_rate_median_smoothed",
    "sciona.atoms.expansion.signal_event_rate.estimate_event_rate_from_signal",
    "sciona.atoms.expansion.signal_event_rate.remove_signal_jumps",
    "sciona.atoms.expansion.signal_event_rate.reject_outlier_intervals",
}


def test_pubrev013_safe_rows_are_provider_owned_and_catalog_ready() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    rows = {row["atom_key"]: row for row in bundle["rows"]}

    assert bundle["provider_repo"] == "sciona-atoms-signal"
    assert SAFE_PUBREV_013_KEYS <= set(rows)

    for atom_key in SAFE_PUBREV_013_KEYS:
        row = rows[atom_key]
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["limitations"]
        assert row["review_record_path"] == "data/review_bundles/signal_expansion.review_bundle.json"
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists()


def test_pubrev013_safe_rows_have_references_and_probe_metadata() -> None:
    references = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    declaration_fqdns = {fqdn for fqdn, _, _ in SIGNAL_EVENT_RATE_DECLARATIONS.values()}
    probe_fqdns = {target.atom_fqdn for target in SIGNAL_EVENT_RATE_PROBE_TARGETS}

    assert SAFE_PUBREV_013_KEYS <= set(references["atoms"])
    assert SAFE_PUBREV_013_KEYS <= declaration_fqdns
    assert SAFE_PUBREV_013_KEYS <= probe_fqdns

    for atom_key in SAFE_PUBREV_013_KEYS:
        entry = references["atoms"][atom_key]
        ref_ids = {ref["ref_id"] for ref in entry["references"]}
        assert {"feli2023pipeline", "scipy2020"} <= ref_ids
        assert entry["auto_attribution_runs"] == []


def test_pubrev013_smoothed_rates_match_source_semantics() -> None:
    events = np.array([0, 10, 20, 40, 50], dtype=np.int64)
    raw_midpoints, raw_rate = compute_event_rate(events, 10.0)

    midpoints, moving = compute_event_rate_smoothed(events, 10.0, smoothing_window=3)
    expected_moving = np.convolve(raw_rate, np.ones(3, dtype=np.float64) / 3.0, mode="same")
    assert midpoints.tolist() == raw_midpoints.tolist()
    np.testing.assert_allclose(moving, expected_moving)

    midpoints, median = compute_event_rate_median_smoothed(events, 10.0, smoothing_window=4)
    expected_median = median_filter(raw_rate.astype(np.float64), size=3, mode="nearest")
    assert midpoints.tolist() == raw_midpoints.tolist()
    np.testing.assert_allclose(median, expected_median)


def test_pubrev013_detection_quality_and_estimator_behaviors() -> None:
    signal = np.zeros(80, dtype=float)
    signal[[10, 30, 50]] = [5.0, 6.0, 5.0]
    assert detect_peaks_in_signal(signal, 20.0, prominence_scale=0.5, refractory_scale=0.2).tolist() == [
        10,
        30,
        50,
    ]

    values, mask = assess_signal_quality(
        np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 10.0]),
        4.0,
        window_seconds=1.0,
        min_kurtosis=-1.0,
    )
    assert values.shape == mask.shape
    assert mask[:4].tolist() == [False, False, False, False]
    assert mask[4:].tolist() == [True, True, True, True]

    sampling_rate = 100.0
    time = np.arange(0.0, 6.0, 1.0 / sampling_rate)
    waveform = np.sin(2.0 * np.pi * 5.0 * time)
    events, midpoints, event_rate = estimate_event_rate_from_signal(waveform, sampling_rate)
    assert events.dtype == np.int64
    assert np.all(np.diff(events) >= 0)
    assert len(midpoints) == len(event_rate)
    assert np.isfinite(event_rate).all()


def test_pubrev013_remediated_helpers_handle_zero_mad_edge_cases() -> None:
    stepped = np.r_[np.zeros(5), np.ones(5) * 10.0]
    np.testing.assert_allclose(remove_signal_jumps(stepped, 10.0), np.zeros_like(stepped))

    noisy_step = np.array([0.0, 0.1, -0.1, 10.0, 10.1, 9.9])
    corrected = remove_signal_jumps(noisy_step, 10.0, jump_threshold_scale=4.0)
    assert np.max(corrected) - np.min(corrected) < 0.5

    single_extra_event = np.array([0, 100, 110, 200, 300], dtype=np.int64)
    np.testing.assert_array_equal(
        reject_outlier_intervals(single_extra_event, 100.0),
        np.array([0, 100, 200, 300], dtype=np.int64),
    )

    zero_mad_extra_event = np.array([0, 100, 110, 200, 300, 400], dtype=np.int64)
    np.testing.assert_array_equal(
        reject_outlier_intervals(zero_mad_extra_event, 100.0),
        np.array([0, 100, 200, 300, 400], dtype=np.int64),
    )

    midpoint_extra_event = np.array([0, 100, 150, 200, 300], dtype=np.int64)
    np.testing.assert_array_equal(
        reject_outlier_intervals(midpoint_extra_event, 100.0),
        np.array([0, 100, 200, 300], dtype=np.int64),
    )

    missed_event_gap = np.array([0, 100, 300, 400], dtype=np.int64)
    np.testing.assert_array_equal(
        reject_outlier_intervals(missed_event_gap, 100.0),
        missed_event_gap,
    )
