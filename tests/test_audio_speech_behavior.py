"""Behavioral tests for the audio_speech atom family."""

from __future__ import annotations

import numpy as np

from sciona.atoms.audio_speech import (
    apply_mel_filterbank,
    audio_windows,
    ctc_beam_decode,
    ctc_greedy_decode,
    dtw_alignment,
    ebu_r128_normalize,
    log_mel_spectrogram,
    median_filter_1d,
    mel_filterbank,
    mfcc,
    monotonic_alignment_search,
    resample_audio,
    rule_based_g2p,
    spec_augment_freq_mask,
    spec_augment_time_mask,
    stft_magnitude,
    wiener_soft_mask,
)


def test_resample_audio_and_windows_shapes() -> None:
    signal = np.ones(100, dtype=np.float64)

    resampled = resample_audio(signal, 100, 200)
    windows = audio_windows(signal, 10, 5)

    assert resampled.shape == (200,)
    assert np.isfinite(resampled).all()
    assert windows.shape == (19, 10)
    assert not windows.flags.writeable


def test_spectral_pipeline_shapes_and_finite_values() -> None:
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 256, dtype=np.float64))

    magnitude = stft_magnitude(signal, n_fft=64, hop_length=32, window="hann")
    power = magnitude * magnitude
    filters = mel_filterbank(n_mels=12, n_fft=64, sr=8000, fmin=20.0, fmax=3800.0)
    mel = apply_mel_filterbank(power, filters)
    log_mel = log_mel_spectrogram(mel, ref=1.0, amin=1e-10, top_db=80.0)
    coeffs = mfcc(log_mel, n_mfcc=5)

    assert magnitude.shape == (33, 7)
    assert filters.shape == (12, 33)
    assert mel.shape == (12, 7)
    assert np.isfinite(log_mel).all()
    assert float(log_mel.max() - log_mel.min()) <= 80.0 + 1e-9
    assert coeffs.shape == (5, 7)


def test_specaugment_is_seeded_and_axis_specific() -> None:
    spectrogram = np.ones((6, 8), dtype=np.float64)

    time_masked = spec_augment_time_mask(spectrogram, 2, 2, 0.0, 7)
    time_masked_again = spec_augment_time_mask(spectrogram, 2, 2, 0.0, 7)
    freq_masked = spec_augment_freq_mask(spectrogram, 2, 2, -1.0, 7)

    assert np.array_equal(time_masked, time_masked_again)
    assert time_masked.shape == spectrogram.shape
    assert freq_masked.shape == spectrogram.shape
    assert np.any(time_masked == 0.0)
    assert np.any(freq_masked == -1.0)


def test_median_filter_removes_single_frame_dropout() -> None:
    predictions = np.array([1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)

    filtered = median_filter_1d(predictions, 3)

    assert np.array_equal(filtered, np.ones_like(predictions))


def test_ebu_r128_normalize_scales_quiet_signal() -> None:
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    quiet = 1.0e-3 * np.sin(2.0 * np.pi * 1000.0 * t)

    normalized = ebu_r128_normalize(quiet, sr, -23.0)

    assert normalized.shape == quiet.shape
    assert np.isfinite(normalized).all()
    assert np.max(np.abs(normalized)) > np.max(np.abs(quiet))


def test_wiener_soft_mask_uses_half_mask_for_equal_sources() -> None:
    target = np.ones((3, 4), dtype=np.float64)
    noise = np.ones((3, 4), dtype=np.float64)
    mixture = np.ones((3, 4), dtype=np.complex128) * (2.0 + 2.0j)

    masked = wiener_soft_mask(target, noise, mixture, eps=1e-12)

    assert masked.shape == mixture.shape
    assert np.allclose(masked, mixture * 0.5, atol=1e-9)


def test_rule_based_g2p_preserves_source_spans() -> None:
    ruleset = {"ph": (102, "", ""), "a": (1, "", "")}

    tokens = rule_based_g2p("phase", ruleset)

    assert tokens[0] == (102, 0, 1)
    assert tokens[1] == (1, 2, 2)
    assert all(start <= end for _, start, end in tokens)


def test_dtw_alignment_tracks_ordered_tokens() -> None:
    frame_probs = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )

    path = dtw_alignment(frame_probs, [0, 1, 2])

    assert path[0] == (0, 0)
    assert path[-1] == (3, 2)
    assert all(left[0] < right[0] and left[1] <= right[1] for left, right in zip(path, path[1:]))


def test_monotonic_alignment_search_returns_one_path_per_frame() -> None:
    value = np.array([[[3.0, 2.0, 1.0], [0.0, 2.0, 3.0]]], dtype=np.float64)
    mask = np.ones_like(value)

    alignment = monotonic_alignment_search(value, mask, -1.0e9)

    assert alignment.shape == value.shape
    assert np.all((alignment == 0.0) | (alignment == 1.0))
    assert np.all(alignment.sum(axis=1) == 1.0)


def test_ctc_decoders_remove_blanks_and_repeats() -> None:
    probs = np.array(
        [
            [0.05, 0.9, 0.05],
            [0.05, 0.9, 0.05],
            [0.95, 0.025, 0.025],
            [0.05, 0.05, 0.9],
            [0.05, 0.05, 0.9],
        ],
        dtype=np.float64,
    )
    log_probs = np.log(probs)

    greedy = ctc_greedy_decode(log_probs, blank_id=0)
    beams = ctc_beam_decode(log_probs, beam_width=3, blank_id=0, alpha=0.0, beta=0.0)

    assert greedy == [1, 2]
    assert beams[0][0] == [1, 2]
    assert len(beams) <= 3
