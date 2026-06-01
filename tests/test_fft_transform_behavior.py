"""Behavioral tests for the fft_transform atom family."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.signal.spectral.fft_transform.atoms import (
    apply_spectral_window,
    optimize_fft_length,
    compute_forward_rfft,
)
from sciona.atoms.signal.spectral.fft_transform.witnesses import (
    witness_apply_spectral_window,
    witness_optimize_fft_length,
    witness_compute_forward_rfft,
)
from sciona.ghost.abstract import AbstractSignal, AbstractScalar, AbstractArray


def test_apply_spectral_window() -> None:
    signal = np.ones(64, dtype=np.float64)
    windowed = apply_spectral_window(signal, "hann")

    assert windowed.shape == signal.shape
    # Symmetric Hann window starts and ends very close to zero
    assert windowed[0] < 1e-2
    assert windowed[-1] < 1e-2
    # Peak at center
    assert np.allclose(windowed[32], 1.0, atol=1e-7)
    assert np.all(windowed >= 0.0)
    assert np.all(windowed <= 1.0)


def test_apply_spectral_window_invalid() -> None:
    signal = np.ones(64, dtype=np.float64)
    with pytest.raises(Exception):
        apply_spectral_window(signal, "non_existent_window")


def test_optimize_fft_length() -> None:
    assert optimize_fft_length(7) >= 7
    assert optimize_fft_length(100) == 100
    assert optimize_fft_length(101) >= 101
    assert optimize_fft_length(12) == 12


def test_compute_forward_rfft() -> None:
    signal = np.sin(np.linspace(0.0, 2.0 * np.pi, 64, dtype=np.float64))
    n = 64
    spectrum = compute_forward_rfft(signal, n)

    assert spectrum.shape == (33,)
    assert np.iscomplexobj(spectrum)
    assert np.all(np.isfinite(spectrum))


def test_witness_apply_spectral_window() -> None:
    signal = AbstractSignal(shape=(64,), dtype="float64", sampling_rate=100.0, domain="time")
    window_type = AbstractScalar(dtype="str")
    witness_out = witness_apply_spectral_window(signal, window_type)

    assert isinstance(witness_out, AbstractSignal)
    assert witness_out.shape == (64,)
    assert witness_out.dtype == "float64"
    assert witness_out.sampling_rate == 100.0


def test_witness_optimize_fft_length() -> None:
    target_len = AbstractScalar(dtype="int64", min_val=10)
    witness_out = witness_optimize_fft_length(target_len)

    assert isinstance(witness_out, AbstractScalar)
    assert witness_out.dtype == "int64"
    assert witness_out.min_val == 10


def test_witness_compute_forward_rfft() -> None:
    signal = AbstractSignal(shape=(64,), dtype="float64", sampling_rate=100.0, domain="time")
    n = AbstractScalar(dtype="int64", min_val=64)
    witness_out = witness_compute_forward_rfft(signal, n)

    assert isinstance(witness_out, AbstractSignal)
    assert witness_out.shape == (33,)
    assert witness_out.dtype == "complex128"
    assert witness_out.sampling_rate == 100.0
    assert witness_out.domain == "freq"
