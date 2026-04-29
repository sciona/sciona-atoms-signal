"""Pure audio and speech processing atoms from the 05 research bundle."""

from __future__ import annotations

import math
import re
from collections import defaultdict

import icontract
import numpy as np
from numpy.typing import NDArray
import scipy.fft
import scipy.signal

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_apply_mel_filterbank,
    witness_audio_windows,
    witness_ctc_beam_decode,
    witness_ctc_greedy_decode,
    witness_dtw_alignment,
    witness_ebu_r128_normalize,
    witness_log_mel_spectrogram,
    witness_median_filter_1d,
    witness_mel_filterbank,
    witness_mfcc,
    witness_monotonic_alignment_search,
    witness_resample_audio,
    witness_rule_based_g2p,
    witness_spec_augment_freq_mask,
    witness_spec_augment_time_mask,
    witness_stft_magnitude,
    witness_wiener_soft_mask,
)


_NEG_INF = -1.0e30
_K_WEIGHTING_B1 = np.array([1.5351248, -2.6916961, 1.1983928], dtype=np.float64)
_K_WEIGHTING_A1 = np.array([1.0, -1.6906592, 0.7324807], dtype=np.float64)
_K_WEIGHTING_B2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
_K_WEIGHTING_A2 = np.array([1.0, -1.9900474, 0.9900722], dtype=np.float64)


def _is_finite(values: NDArray[np.float64]) -> bool:
    return bool(np.isfinite(values).all())


def _is_finite_or_complex(values: NDArray[np.complex128]) -> bool:
    return bool(np.isfinite(values.real).all() and np.isfinite(values.imag).all())


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _logsumexp_pair(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high = max(left, right)
    return high + math.log(math.exp(left - high) + math.exp(right - high))


def _logsumexp_many(values: list[float]) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    high = max(finite)
    return high + math.log(sum(math.exp(value - high) for value in finite))


def _hz_to_mel_slaney(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    min_log_hz = 1000.0
    min_log_mel = 15.0
    linear_step = 200.0 / 3.0
    log_step = math.log(6.4) / 27.0
    mels = frequencies / linear_step
    log_region = frequencies >= min_log_hz
    mels[log_region] = min_log_mel + np.log(frequencies[log_region] / min_log_hz) / log_step
    return mels


def _mel_to_hz_slaney(mels: NDArray[np.float64]) -> NDArray[np.float64]:
    min_log_hz = 1000.0
    min_log_mel = 15.0
    linear_step = 200.0 / 3.0
    log_step = math.log(6.4) / 27.0
    frequencies = mels * linear_step
    log_region = mels >= min_log_mel
    frequencies[log_region] = min_log_hz * np.exp(log_step * (mels[log_region] - min_log_mel))
    return frequencies


@register_atom(witness_resample_audio)
@icontract.require(lambda signal: signal.ndim in (1, 2), "signal must be 1-D or 2-D")
@icontract.require(lambda signal: _is_finite(signal), "signal must contain finite samples")
@icontract.require(lambda orig_sr: orig_sr > 0, "orig_sr must be positive")
@icontract.require(lambda target_sr: target_sr > 0, "target_sr must be positive")
@icontract.ensure(lambda result, signal, orig_sr, target_sr: result.shape[-1] == math.ceil(signal.shape[-1] * target_sr / orig_sr), "sample count must follow the sample-rate ratio")
@icontract.ensure(lambda result: _is_finite(result), "resampled signal must be finite")
def resample_audio(
    signal: NDArray[np.float64],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float64]:
    """Resample audio with polyphase anti-alias filtering along the sample axis."""
    divisor = math.gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    return scipy.signal.resample_poly(signal, up, down, axis=-1).astype(np.float64)


@register_atom(witness_audio_windows)
@icontract.require(lambda signal: signal.ndim == 1, "signal must be 1-D")
@icontract.require(lambda signal: _is_finite(signal), "signal must contain finite samples")
@icontract.require(lambda window_size: window_size > 0, "window_size must be positive")
@icontract.require(lambda hop_size: hop_size > 0, "hop_size must be positive")
@icontract.require(lambda signal, window_size: signal.shape[0] >= window_size, "signal must cover at least one window")
@icontract.ensure(lambda result, signal, window_size, hop_size: result.shape == ((signal.shape[0] - window_size) // hop_size + 1, window_size), "window grid shape must match the requested frame geometry")
@icontract.ensure(lambda result: not result.flags.writeable, "window view must be read-only")
def audio_windows(
    signal: NDArray[np.float64],
    window_size: int,
    hop_size: int,
) -> NDArray[np.float64]:
    """Expose overlapping audio windows as a read-only strided view."""
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::hop_size]
    windows.setflags(write=False)
    return windows.astype(np.float64, copy=False)


@register_atom(witness_stft_magnitude)
@icontract.require(lambda signal: signal.ndim == 1, "signal must be 1-D")
@icontract.require(lambda signal: _is_finite(signal), "signal must contain finite samples")
@icontract.require(lambda n_fft: _is_power_of_two(n_fft), "n_fft must be a positive power of two")
@icontract.require(lambda hop_length: hop_length > 0, "hop_length must be positive")
@icontract.require(lambda window: window in {"hann", "boxcar"}, "window must be hann or boxcar")
@icontract.require(lambda signal, n_fft: signal.shape[0] >= n_fft, "signal must cover one FFT frame")
@icontract.ensure(lambda result, n_fft: result.shape[0] == 1 + n_fft // 2, "frequency-bin count must match rfft")
@icontract.ensure(lambda result: _is_finite(result), "magnitude spectrum must be finite")
@icontract.ensure(lambda result: np.all(result >= 0.0), "magnitudes must be non-negative")
def stft_magnitude(
    signal: NDArray[np.float64],
    n_fft: int,
    hop_length: int,
    window: str,
) -> NDArray[np.float64]:
    """Compute a one-sided STFT magnitude spectrogram from framed audio."""
    frames = audio_windows(signal, n_fft, hop_length)
    taper = np.hanning(n_fft) if window == "hann" else np.ones(n_fft, dtype=np.float64)
    spectrum = np.fft.rfft(frames * taper, n=n_fft, axis=1)
    return np.abs(spectrum).T.astype(np.float64)


@register_atom(witness_mel_filterbank)
@icontract.require(lambda sr: sr > 0, "sample rate must be positive")
@icontract.require(lambda n_mels: n_mels > 0, "n_mels must be positive")
@icontract.require(lambda n_fft: n_fft > 0, "n_fft must be positive")
@icontract.require(lambda sr, fmin, fmax: 0.0 <= fmin < fmax <= sr / 2.0, "mel frequency bounds must fit Nyquist")
@icontract.ensure(lambda result, n_mels, n_fft: result.shape == (n_mels, 1 + n_fft // 2), "filterbank shape must match mel and FFT bins")
@icontract.ensure(lambda result: _is_finite(result), "filterbank weights must be finite")
@icontract.ensure(lambda result: np.all(result >= 0.0), "filterbank weights must be non-negative")
def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sr: int,
    fmin: float,
    fmax: float,
) -> NDArray[np.float64]:
    """Build a Slaney-normalized triangular mel filterbank."""
    fft_freqs = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2, dtype=np.float64)
    mel_edges = np.linspace(
        _hz_to_mel_slaney(np.array([fmin], dtype=np.float64))[0],
        _hz_to_mel_slaney(np.array([fmax], dtype=np.float64))[0],
        n_mels + 2,
        dtype=np.float64,
    )
    hz_edges = _mel_to_hz_slaney(mel_edges)
    ramps = hz_edges[:, np.newaxis] - fft_freqs[np.newaxis, :]
    weights = np.zeros((n_mels, fft_freqs.size), dtype=np.float64)
    for band in range(n_mels):
        lower = -ramps[band] / max(hz_edges[band + 1] - hz_edges[band], np.finfo(float).eps)
        upper = ramps[band + 2] / max(hz_edges[band + 2] - hz_edges[band + 1], np.finfo(float).eps)
        weights[band] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / np.maximum(hz_edges[2 : n_mels + 2] - hz_edges[:n_mels], np.finfo(float).eps)
    return (weights * enorm[:, np.newaxis]).astype(np.float64)


@register_atom(witness_apply_mel_filterbank)
@icontract.require(lambda power_spectrum: power_spectrum.ndim == 2, "power_spectrum must be 2-D")
@icontract.require(lambda mel_fb: mel_fb.ndim == 2, "mel_fb must be 2-D")
@icontract.require(lambda power_spectrum, mel_fb: power_spectrum.shape[0] == mel_fb.shape[1], "FFT-bin axes must match")
@icontract.require(lambda power_spectrum, mel_fb: _is_finite(power_spectrum) and _is_finite(mel_fb), "inputs must be finite")
@icontract.require(lambda power_spectrum, mel_fb: bool(np.all(power_spectrum >= 0.0) and np.all(mel_fb >= 0.0)), "power and filter weights must be non-negative")
@icontract.ensure(lambda result, power_spectrum, mel_fb: result.shape == (mel_fb.shape[0], power_spectrum.shape[1]), "mel projection shape must match bands and frames")
@icontract.ensure(lambda result: _is_finite(result), "mel spectrum must be finite")
@icontract.ensure(lambda result: np.all(result >= 0.0), "mel spectrum must be non-negative")
def apply_mel_filterbank(
    power_spectrum: NDArray[np.float64],
    mel_fb: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project a linear power spectrogram onto mel bands."""
    return np.dot(mel_fb, power_spectrum).astype(np.float64)


@register_atom(witness_log_mel_spectrogram)
@icontract.require(lambda mel_spectrum: mel_spectrum.ndim == 2, "mel_spectrum must be 2-D")
@icontract.require(lambda mel_spectrum: _is_finite(mel_spectrum), "mel_spectrum must be finite")
@icontract.require(lambda mel_spectrum: bool(np.all(mel_spectrum >= 0.0)), "mel_spectrum must be non-negative")
@icontract.require(lambda ref: ref > 0.0, "ref must be positive")
@icontract.require(lambda amin: amin > 0.0, "amin must be positive")
@icontract.require(lambda top_db: top_db > 0.0, "top_db must be positive")
@icontract.ensure(lambda result, mel_spectrum: result.shape == mel_spectrum.shape, "log scaling must preserve shape")
@icontract.ensure(lambda result: _is_finite(result), "log mel values must be finite")
@icontract.ensure(lambda result, top_db: float(np.max(result) - np.min(result)) <= top_db + 1e-9, "dynamic range must be clipped")
def log_mel_spectrogram(
    mel_spectrum: NDArray[np.float64],
    ref: float,
    amin: float,
    top_db: float,
) -> NDArray[np.float64]:
    """Convert mel power values to a clipped decibel scale."""
    log_spec = 10.0 * np.log10(np.maximum(amin, mel_spectrum))
    log_spec -= 10.0 * math.log10(max(amin, ref))
    return np.maximum(log_spec, float(np.max(log_spec)) - top_db).astype(np.float64)


@register_atom(witness_mfcc)
@icontract.require(lambda log_mel_spectrum: log_mel_spectrum.ndim == 2, "log_mel_spectrum must be 2-D")
@icontract.require(lambda log_mel_spectrum: _is_finite(log_mel_spectrum), "log_mel_spectrum must be finite")
@icontract.require(lambda log_mel_spectrum, n_mfcc: 0 < n_mfcc <= log_mel_spectrum.shape[0], "n_mfcc must fit the mel axis")
@icontract.ensure(lambda result, log_mel_spectrum, n_mfcc: result.shape == (n_mfcc, log_mel_spectrum.shape[1]), "MFCC shape must match requested coefficients")
@icontract.ensure(lambda result: _is_finite(result), "MFCC values must be finite")
def mfcc(
    log_mel_spectrum: NDArray[np.float64],
    n_mfcc: int,
) -> NDArray[np.float64]:
    """Compute DCT-II cepstral coefficients from a log-mel spectrogram."""
    cepstra = scipy.fft.dct(log_mel_spectrum, type=2, axis=0, norm="ortho")
    return cepstra[:n_mfcc].astype(np.float64)


@register_atom(witness_spec_augment_time_mask)
@icontract.require(lambda spectrogram: spectrogram.ndim == 2, "spectrogram must be 2-D")
@icontract.require(lambda spectrogram: _is_finite(spectrogram), "spectrogram must be finite")
@icontract.require(lambda num_masks: num_masks >= 0, "num_masks must be non-negative")
@icontract.require(lambda max_width: max_width > 0, "max_width must be positive")
@icontract.ensure(lambda result, spectrogram: result.shape == spectrogram.shape, "masking must preserve shape")
@icontract.ensure(lambda result: _is_finite(result), "masked spectrogram must be finite")
def spec_augment_time_mask(
    spectrogram: NDArray[np.float64],
    num_masks: int,
    max_width: int,
    mask_value: float,
    rng_seed: int,
) -> NDArray[np.float64]:
    """Apply deterministic SpecAugment masks along the time axis."""
    result = spectrogram.copy()
    rng = np.random.default_rng(rng_seed)
    time_bins = result.shape[1]
    max_span = min(max_width, time_bins)
    for _ in range(num_masks):
        width = int(rng.integers(1, max_span + 1))
        start = int(rng.integers(0, time_bins - width + 1))
        result[:, start : start + width] = mask_value
    return result.astype(np.float64)


@register_atom(witness_spec_augment_freq_mask)
@icontract.require(lambda spectrogram: spectrogram.ndim == 2, "spectrogram must be 2-D")
@icontract.require(lambda spectrogram: _is_finite(spectrogram), "spectrogram must be finite")
@icontract.require(lambda num_masks: num_masks >= 0, "num_masks must be non-negative")
@icontract.require(lambda max_width: max_width > 0, "max_width must be positive")
@icontract.ensure(lambda result, spectrogram: result.shape == spectrogram.shape, "masking must preserve shape")
@icontract.ensure(lambda result: _is_finite(result), "masked spectrogram must be finite")
def spec_augment_freq_mask(
    spectrogram: NDArray[np.float64],
    num_masks: int,
    max_width: int,
    mask_value: float,
    rng_seed: int,
) -> NDArray[np.float64]:
    """Apply deterministic SpecAugment masks along the frequency axis."""
    result = spectrogram.copy()
    rng = np.random.default_rng(rng_seed)
    freq_bins = result.shape[0]
    max_span = min(max_width, freq_bins)
    for _ in range(num_masks):
        width = int(rng.integers(1, max_span + 1))
        start = int(rng.integers(0, freq_bins - width + 1))
        result[start : start + width, :] = mask_value
    return result.astype(np.float64)


@register_atom(witness_median_filter_1d)
@icontract.require(lambda predictions: predictions.ndim == 1, "predictions must be 1-D")
@icontract.require(lambda predictions: _is_finite(predictions), "predictions must be finite")
@icontract.require(lambda kernel_size: kernel_size > 0 and kernel_size % 2 == 1, "kernel_size must be a positive odd integer")
@icontract.ensure(lambda result, predictions: result.shape == predictions.shape, "median filtering must preserve shape")
@icontract.ensure(lambda result: _is_finite(result), "filtered predictions must be finite")
def median_filter_1d(
    predictions: NDArray[np.float64],
    kernel_size: int,
) -> NDArray[np.float64]:
    """Smooth a frame-level prediction sequence with a centered median filter."""
    radius = kernel_size // 2
    padded = np.pad(predictions, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
    return np.median(windows, axis=1).astype(np.float64)


@register_atom(witness_ebu_r128_normalize)
@icontract.require(lambda signal: signal.ndim in (1, 2), "signal must be 1-D or 2-D")
@icontract.require(lambda signal: _is_finite(signal), "signal must contain finite samples")
@icontract.require(lambda signal: signal.shape[-1] >= 19200, "signal must contain at least one 400 ms block at 48 kHz")
@icontract.require(lambda sr: sr == 48000, "sr must be 48000 for the fixed BS.1770 coefficient path")
@icontract.require(lambda target_lufs: target_lufs <= 0.0, "target_lufs must be non-positive")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "normalization must preserve shape")
@icontract.ensure(lambda result: _is_finite(result), "normalized signal must be finite")
def ebu_r128_normalize(
    signal: NDArray[np.float64],
    sr: int,
    target_lufs: float,
) -> NDArray[np.float64]:
    """Normalize 48 kHz audio using K-weighted gated integrated loudness."""
    weighted = scipy.signal.lfilter(_K_WEIGHTING_B1, _K_WEIGHTING_A1, signal, axis=-1)
    weighted = scipy.signal.lfilter(_K_WEIGHTING_B2, _K_WEIGHTING_A2, weighted, axis=-1)
    channels = weighted[np.newaxis, :] if weighted.ndim == 1 else weighted
    block = int(round(0.400 * sr))
    hop = block // 4
    energies = []
    for start in range(0, channels.shape[-1] - block + 1, hop):
        segment = channels[..., start : start + block]
        energies.append(float(np.mean(np.sum(segment * segment, axis=0))))
    block_energy = np.array(energies, dtype=np.float64)
    loudness = -0.691 + 10.0 * np.log10(np.maximum(block_energy, np.finfo(float).tiny))
    gated = block_energy[loudness > -70.0]
    if gated.size == 0:
        return signal.copy().astype(np.float64)
    relative_loudness = -0.691 + 10.0 * math.log10(float(np.mean(gated)))
    gated = gated[loudness[loudness > -70.0] > relative_loudness - 10.0]
    if gated.size == 0:
        return signal.copy().astype(np.float64)
    integrated = -0.691 + 10.0 * math.log10(float(np.mean(gated)))
    gain = 10.0 ** ((target_lufs - integrated) / 20.0)
    return (signal * gain).astype(np.float64)


@register_atom(witness_wiener_soft_mask)
@icontract.require(lambda target_mag: target_mag.ndim == 2, "target_mag must be 2-D")
@icontract.require(lambda noise_mag: noise_mag.ndim == 2, "noise_mag must be 2-D")
@icontract.require(lambda mix_complex: mix_complex.ndim == 2, "mix_complex must be 2-D")
@icontract.require(lambda target_mag, noise_mag, mix_complex: target_mag.shape == noise_mag.shape == mix_complex.shape, "spectrogram shapes must match")
@icontract.require(lambda target_mag, noise_mag: _is_finite(target_mag) and _is_finite(noise_mag), "magnitude estimates must be finite")
@icontract.require(lambda mix_complex: _is_finite_or_complex(mix_complex), "mixture spectrogram must be finite")
@icontract.require(lambda target_mag, noise_mag: bool(np.all(target_mag >= 0.0) and np.all(noise_mag >= 0.0)), "magnitudes must be non-negative")
@icontract.require(lambda eps: eps > 0.0, "eps must be positive")
@icontract.ensure(lambda result, mix_complex: result.shape == mix_complex.shape, "masked spectrogram must preserve shape")
@icontract.ensure(lambda result: _is_finite_or_complex(result), "masked spectrogram must be finite")
def wiener_soft_mask(
    target_mag: NDArray[np.float64],
    noise_mag: NDArray[np.float64],
    mix_complex: NDArray[np.complex128],
    eps: float,
) -> NDArray[np.complex128]:
    """Apply a stabilized power-ratio soft mask to a complex mixture spectrum."""
    target_power = target_mag * target_mag
    noise_power = noise_mag * noise_mag
    mask = target_power / (target_power + noise_power + eps)
    return (mix_complex * mask).astype(np.complex128)


@register_atom(witness_rule_based_g2p)
@icontract.require(lambda text: len(text) > 0, "text must be non-empty")
@icontract.require(lambda ruleset: len(ruleset) > 0, "ruleset must be non-empty")
@icontract.ensure(lambda result: len(result) > 0, "at least one phoneme token must be emitted")
@icontract.ensure(lambda result: all(start <= end for _, start, end in result), "source spans must be ordered")
def rule_based_g2p(
    text: str,
    ruleset: dict[str, tuple[int, str, str]],
) -> list[tuple[int, int, int]]:
    """Convert text to phoneme IDs using explicit regex context rules.

    The ruleset maps a grapheme pattern to ``(phoneme_id, left_regex,
    right_regex)``.  Longest matching grapheme patterns are tried first, and
    unmatched characters emit their Unicode code point as a deterministic
    fallback token.
    """
    ordered_rules = sorted(ruleset.items(), key=lambda item: len(item[0]), reverse=True)
    result: list[tuple[int, int, int]] = []
    idx = 0
    while idx < len(text):
        matched = False
        for grapheme, (phoneme_id, left_regex, right_regex) in ordered_rules:
            if not text.startswith(grapheme, idx):
                continue
            left_ok = re.search(f"{left_regex}$", text[:idx]) is not None if left_regex else True
            right_start = idx + len(grapheme)
            right_ok = re.match(right_regex, text[right_start:]) is not None if right_regex else True
            if left_ok and right_ok:
                result.append((int(phoneme_id), idx, right_start - 1))
                idx = right_start
                matched = True
                break
        if not matched:
            result.append((ord(text[idx]), idx, idx))
            idx += 1
    return result


@register_atom(witness_dtw_alignment)
@icontract.require(lambda frame_probs: frame_probs.ndim == 2, "frame_probs must be 2-D")
@icontract.require(lambda frame_probs: _is_finite(frame_probs), "frame_probs must be finite")
@icontract.require(lambda frame_probs: bool(np.all(frame_probs >= 0.0)), "frame_probs must be non-negative")
@icontract.require(lambda frame_probs, phoneme_sequence: frame_probs.shape[0] >= len(phoneme_sequence) > 0, "there must be at least one frame per target token")
@icontract.require(lambda frame_probs, phoneme_sequence: all(0 <= token < frame_probs.shape[1] for token in phoneme_sequence), "phoneme IDs must fit the probability axis")
@icontract.ensure(lambda result: result[0] == (0, 0), "alignment must start at the first frame and token")
@icontract.ensure(lambda result, frame_probs, phoneme_sequence: result[-1] == (frame_probs.shape[0] - 1, len(phoneme_sequence) - 1), "alignment must end at the final frame and token")
@icontract.ensure(lambda result: all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(result, result[1:])), "alignment must advance monotonically")
def dtw_alignment(
    frame_probs: NDArray[np.float64],
    phoneme_sequence: list[int],
) -> list[tuple[int, int]]:
    """Find a monotonic frame-to-token DTW path from frame probabilities."""
    costs = -np.log(np.maximum(frame_probs[:, phoneme_sequence], np.finfo(float).tiny))
    frames, tokens = costs.shape
    dp = np.full((frames, tokens), np.inf, dtype=np.float64)
    back = np.zeros((frames, tokens), dtype=np.int64)
    dp[0, 0] = costs[0, 0]
    for frame in range(1, frames):
        dp[frame, 0] = dp[frame - 1, 0] + costs[frame, 0]
    for frame in range(1, frames):
        max_token = min(frame, tokens - 1)
        for token in range(1, max_token + 1):
            stay = dp[frame - 1, token]
            advance = dp[frame - 1, token - 1]
            if advance <= stay:
                dp[frame, token] = advance + costs[frame, token]
                back[frame, token] = token - 1
            else:
                dp[frame, token] = stay + costs[frame, token]
                back[frame, token] = token
    path: list[tuple[int, int]] = []
    token = tokens - 1
    for frame in range(frames - 1, -1, -1):
        path.append((frame, token))
        if frame > 0:
            token = int(back[frame, token])
    path.reverse()
    return path


@register_atom(witness_monotonic_alignment_search)
@icontract.require(lambda value: value.ndim == 3, "value must be 3-D")
@icontract.require(lambda mask: mask.ndim == 3, "mask must be 3-D")
@icontract.require(lambda value, mask: value.shape == mask.shape, "value and mask shapes must match")
@icontract.require(lambda value, mask: _is_finite(value) and _is_finite(mask), "value and mask must be finite")
@icontract.require(lambda max_neg_val: max_neg_val < 0.0, "max_neg_val must be negative")
@icontract.ensure(lambda result, value: result.shape == value.shape, "alignment mask must preserve shape")
@icontract.ensure(lambda result: bool(np.all((result == 0.0) | (result == 1.0))), "alignment mask must be binary")
def monotonic_alignment_search(
    value: NDArray[np.float64],
    mask: NDArray[np.float64],
    max_neg_val: float,
) -> NDArray[np.float64]:
    """Compute a hard monotonic maximum-score path for each batch item."""
    batch, tokens, frames = value.shape
    result = np.zeros_like(value, dtype=np.float64)
    masked_value = np.where(mask > 0.0, value, max_neg_val)
    for item in range(batch):
        dp = np.full((tokens, frames), max_neg_val, dtype=np.float64)
        ptr = np.zeros((tokens, frames), dtype=bool)
        dp[0, 0] = masked_value[item, 0, 0]
        for frame in range(1, frames):
            max_token = min(frame, tokens - 1)
            for token in range(max_token + 1):
                stay = dp[token, frame - 1]
                advance = dp[token - 1, frame - 1] if token > 0 else max_neg_val
                if advance > stay:
                    ptr[token, frame] = True
                    dp[token, frame] = advance + masked_value[item, token, frame]
                else:
                    dp[token, frame] = stay + masked_value[item, token, frame]
        valid_end = np.flatnonzero(mask[item, :, -1] > 0.0)
        token = int(valid_end[-1]) if valid_end.size else tokens - 1
        for frame in range(frames - 1, -1, -1):
            result[item, token, frame] = 1.0
            if frame > 0 and ptr[token, frame]:
                token = max(0, token - 1)
    return result


@register_atom(witness_ctc_greedy_decode)
@icontract.require(lambda log_probs: log_probs.ndim == 2, "log_probs must be 2-D")
@icontract.require(lambda log_probs: _is_finite(log_probs), "log_probs must be finite")
@icontract.require(lambda log_probs, blank_id: 0 <= blank_id < log_probs.shape[1], "blank_id must fit vocabulary")
@icontract.ensure(lambda result, blank_id: blank_id not in result, "decoded tokens must not include blanks")
def ctc_greedy_decode(
    log_probs: NDArray[np.float64],
    blank_id: int,
) -> list[int]:
    """Collapse the highest-probability CTC path into token IDs."""
    raw = np.argmax(log_probs, axis=1).tolist()
    decoded: list[int] = []
    previous = None
    for token in raw:
        token = int(token)
        if token != previous and token != blank_id:
            decoded.append(token)
        previous = token
    return decoded


@register_atom(witness_ctc_beam_decode)
@icontract.require(lambda log_probs: log_probs.ndim == 2, "log_probs must be 2-D")
@icontract.require(lambda log_probs: _is_finite(log_probs), "log_probs must be finite")
@icontract.require(lambda log_probs, blank_id: 0 <= blank_id < log_probs.shape[1], "blank_id must fit vocabulary")
@icontract.require(lambda beam_width: beam_width > 0, "beam_width must be positive")
@icontract.ensure(lambda result, beam_width: len(result) <= beam_width, "beam output must be pruned")
@icontract.ensure(lambda result, blank_id: all(blank_id not in tokens for tokens, _ in result), "decoded beams must not include blanks")
def ctc_beam_decode(
    log_probs: NDArray[np.float64],
    beam_width: int,
    blank_id: int,
    alpha: float,
    beta: float,
) -> list[tuple[list[int], float]]:
    """Decode CTC log probabilities with prefix beam search in log space."""
    beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
    vocab_size = log_probs.shape[1]
    insertion_bonus = beta
    _ = alpha

    for frame in range(log_probs.shape[0]):
        next_beam: defaultdict[tuple[int, ...], list[float]] = defaultdict(lambda: [-math.inf, -math.inf])
        for prefix, (p_blank, p_not_blank) in beam.items():
            for token in range(vocab_size):
                prob = float(log_probs[frame, token])
                if token == blank_id:
                    entry = next_beam[prefix]
                    entry[0] = _logsumexp_many([entry[0], p_blank + prob, p_not_blank + prob])
                    continue
                end_token = prefix[-1] if prefix else None
                if token == end_token:
                    entry = next_beam[prefix]
                    entry[1] = _logsumexp_pair(entry[1], p_not_blank + prob)
                    new_entry = next_beam[prefix + (token,)]
                    new_entry[1] = _logsumexp_pair(new_entry[1], p_blank + prob + insertion_bonus)
                else:
                    new_entry = next_beam[prefix + (token,)]
                    new_entry[1] = _logsumexp_many(
                        [new_entry[1], p_blank + prob + insertion_bonus, p_not_blank + prob + insertion_bonus]
                    )
        scored = sorted(
            next_beam.items(),
            key=lambda item: _logsumexp_pair(item[1][0], item[1][1]),
            reverse=True,
        )
        beam = {prefix: (scores[0], scores[1]) for prefix, scores in scored[:beam_width]}

    return [
        (list(prefix), _logsumexp_pair(p_blank, p_not_blank))
        for prefix, (p_blank, p_not_blank) in sorted(
            beam.items(),
            key=lambda item: _logsumexp_pair(item[1][0], item[1][1]),
            reverse=True,
        )
    ]
