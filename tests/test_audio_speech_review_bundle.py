from __future__ import annotations

import json
from pathlib import Path


EXPECTED_KEYS = {
    "sciona.atoms.audio_speech.resample_audio",
    "sciona.atoms.audio_speech.audio_windows",
    "sciona.atoms.audio_speech.stft_magnitude",
    "sciona.atoms.audio_speech.mel_filterbank",
    "sciona.atoms.audio_speech.apply_mel_filterbank",
    "sciona.atoms.audio_speech.log_mel_spectrogram",
    "sciona.atoms.audio_speech.mfcc",
    "sciona.atoms.audio_speech.spec_augment_time_mask",
    "sciona.atoms.audio_speech.spec_augment_freq_mask",
    "sciona.atoms.audio_speech.median_filter_1d",
    "sciona.atoms.audio_speech.ebu_r128_normalize",
    "sciona.atoms.audio_speech.wiener_soft_mask",
    "sciona.atoms.audio_speech.rule_based_g2p",
    "sciona.atoms.audio_speech.dtw_alignment",
    "sciona.atoms.audio_speech.monotonic_alignment_search",
    "sciona.atoms.audio_speech.ctc_greedy_decode",
    "sciona.atoms.audio_speech.ctc_beam_decode",
}


def test_audio_speech_review_bundle_structure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_path = repo_root / "data" / "review_bundles" / "audio_speech.review_bundle.json"
    bundle = json.loads(bundle_path.read_text())

    assert bundle["provider_repo"] == "sciona-atoms-signal"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert bundle["review_developer_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert bundle["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}
    assert bundle["review_record_path"] == "data/review_bundles/audio_speech.review_bundle.json"
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_KEYS

    for source in bundle["authoritative_sources"]:
        assert (repo_root / source["path"]).exists()

    for row in bundle["rows"]:
        assert row["atom_name"] == row["atom_key"]
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_status"] == "reviewed"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["review_record_path"] == bundle["review_record_path"]
        for rel_path in row["source_paths"]:
            assert (repo_root / rel_path).exists()
