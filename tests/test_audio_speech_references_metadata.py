from __future__ import annotations

import json
from pathlib import Path


EXPECTED_ATOMS = {
    "resample_audio",
    "audio_windows",
    "stft_magnitude",
    "mel_filterbank",
    "apply_mel_filterbank",
    "log_mel_spectrogram",
    "mfcc",
    "spec_augment_time_mask",
    "spec_augment_freq_mask",
    "median_filter_1d",
    "ebu_r128_normalize",
    "wiener_soft_mask",
    "rule_based_g2p",
    "dtw_alignment",
    "monotonic_alignment_search",
    "ctc_greedy_decode",
    "ctc_beam_decode",
}


def test_audio_speech_references_resolve_to_local_registry() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    refs_path = repo_root / "src" / "sciona" / "atoms" / "audio_speech" / "references.json"
    registry_path = repo_root / "data" / "references" / "registry.json"

    refs = json.loads(refs_path.read_text())
    registry = json.loads(registry_path.read_text())
    registry_ids = set(registry["references"])
    atom_refs = refs["atoms"]
    leaf_names = {key.split("@")[0].rsplit(".", 1)[-1] for key in atom_refs}

    assert refs["schema_version"] == "1.1"
    assert leaf_names == EXPECTED_ATOMS

    for atom_key, entry in atom_refs.items():
        assert atom_key.startswith("sciona.atoms.audio_speech.atoms.")
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"]
            assert metadata["notes"]
