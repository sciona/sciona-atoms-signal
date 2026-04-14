from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_generatereconstructedppg(
    ppg_clean: AbstractSignal,
    noise: AbstractSignal,
    sampling_rate: AbstractScalar,
    generator: AbstractSignal,
    device: AbstractScalar,
) -> AbstractSignal:
    """Describe the reconstructed waveform produced by the GAN generator."""
    return AbstractSignal(
        shape=ppg_clean.shape,
        dtype=ppg_clean.dtype,
        sampling_rate=ppg_clean.sampling_rate,
        domain=ppg_clean.domain,
        units=ppg_clean.units,
    )


def witness_gan_reconstruction(
    ppg_clean: AbstractArray,
    noise: AbstractArray,
    sampling_rate: AbstractScalar,
    generator: AbstractArray,
    device: AbstractScalar,
) -> AbstractArray:
    """Describe the reconstructed sample batch returned by the GAN wrapper."""
    return AbstractArray(shape=(1,) + ppg_clean.shape, dtype="float64")
