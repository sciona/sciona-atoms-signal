"""Ghost witnesses for audio and speech processing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_resample_audio(
    signal: AbstractArray,
    orig_sr: int,
    target_sr: int,
) -> AbstractArray:
    """Resampling changes only the final sample axis."""
    length = (signal.shape[-1] * target_sr + orig_sr - 1) // orig_sr
    return AbstractArray(shape=(*signal.shape[:-1], length), dtype="float64")


def witness_audio_windows(
    signal: AbstractArray,
    window_size: int,
    hop_size: int,
) -> AbstractArray:
    """Windowing maps one time axis into frame and sample axes."""
    frames = (signal.shape[0] - window_size) // hop_size + 1
    return AbstractArray(shape=(frames, window_size), dtype="float64")


def witness_stft_magnitude(
    signal: AbstractArray,
    n_fft: int,
    hop_length: int,
    window: str,
) -> AbstractArray:
    """STFT magnitude returns frequency bins by frame count."""
    frames = (signal.shape[0] - n_fft) // hop_length + 1
    return AbstractArray(shape=(1 + n_fft // 2, frames), dtype="float64")


def witness_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sr: int,
    fmin: float,
    fmax: float,
) -> AbstractArray:
    """Mel filterbanks map FFT bins to mel bands."""
    return AbstractArray(shape=(n_mels, 1 + n_fft // 2), dtype="float64")


def witness_apply_mel_filterbank(
    power_spectrum: AbstractArray,
    mel_fb: AbstractArray,
) -> AbstractArray:
    """Mel projection preserves the time-frame axis."""
    return AbstractArray(shape=(mel_fb.shape[0], power_spectrum.shape[1]), dtype="float64")


def witness_log_mel_spectrogram(
    mel_spectrum: AbstractArray,
    ref: float,
    amin: float,
    top_db: float,
) -> AbstractArray:
    """Log scaling preserves the mel spectrogram shape."""
    return AbstractArray(shape=mel_spectrum.shape, dtype="float64")


def witness_mfcc(
    log_mel_spectrum: AbstractArray,
    n_mfcc: int,
) -> AbstractArray:
    """MFCC truncates the mel-frequency axis."""
    return AbstractArray(shape=(n_mfcc, log_mel_spectrum.shape[1]), dtype="float64")


def witness_spec_augment_time_mask(
    spectrogram: AbstractArray,
    num_masks: int,
    max_width: int,
    mask_value: float,
    rng_seed: int,
) -> AbstractArray:
    """Time masking preserves spectrogram shape."""
    return AbstractArray(shape=spectrogram.shape, dtype="float64")


def witness_spec_augment_freq_mask(
    spectrogram: AbstractArray,
    num_masks: int,
    max_width: int,
    mask_value: float,
    rng_seed: int,
) -> AbstractArray:
    """Frequency masking preserves spectrogram shape."""
    return AbstractArray(shape=spectrogram.shape, dtype="float64")


def witness_median_filter_1d(
    predictions: AbstractArray,
    kernel_size: int,
) -> AbstractArray:
    """Median smoothing preserves the prediction axis."""
    return AbstractArray(shape=predictions.shape, dtype="float64")


def witness_ebu_r128_normalize(
    signal: AbstractArray,
    sr: int,
    target_lufs: float,
) -> AbstractArray:
    """Loudness normalization preserves waveform shape."""
    return AbstractArray(shape=signal.shape, dtype="float64")


def witness_wiener_soft_mask(
    target_mag: AbstractArray,
    noise_mag: AbstractArray,
    mix_complex: AbstractArray,
    eps: float,
) -> AbstractArray:
    """Soft masking preserves the complex spectrogram grid."""
    return AbstractArray(shape=mix_complex.shape, dtype="complex128")


def witness_rule_based_g2p(
    text: str,
    ruleset: dict[str, tuple[int, str, str]],
) -> list[tuple[int, int, int]]:
    """Rule-based G2P emits token IDs with source character spans."""
    return [(0, 0, max(0, len(text) - 1))]


def witness_dtw_alignment(
    frame_probs: AbstractArray,
    phoneme_sequence: list[int],
) -> list[tuple[int, int]]:
    """DTW emits one monotonic token index per frame."""
    return [(idx, min(idx, len(phoneme_sequence) - 1)) for idx in range(frame_probs.shape[0])]


def witness_monotonic_alignment_search(
    value: AbstractArray,
    mask: AbstractArray,
    max_neg_val: float,
) -> AbstractArray:
    """MAS returns a binary path mask with the same shape as the score tensor."""
    return AbstractArray(shape=value.shape, dtype="float64")


def witness_ctc_greedy_decode(
    log_probs: AbstractArray,
    blank_id: int,
) -> list[int]:
    """Greedy CTC returns a collapsed token sequence."""
    return []


def witness_ctc_beam_decode(
    log_probs: AbstractArray,
    beam_width: int,
    blank_id: int,
    alpha: float,
    beta: float,
) -> list[tuple[list[int], float]]:
    """Beam CTC returns sorted token prefixes and log scores."""
    return [([], 0.0)]
