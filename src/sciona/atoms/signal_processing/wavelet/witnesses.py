"""Ghost witnesses for the wavelet denoising atom."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractSignal, AbstractScalar


def witness_wavelet_denoise(
    signal: AbstractSignal,
    wavelet: AbstractScalar,
    level: AbstractScalar,
    threshold_mode: AbstractScalar,
) -> AbstractSignal:
    """Denoising preserves signal shape, sampling rate, and domain."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain=signal.domain,
    )
