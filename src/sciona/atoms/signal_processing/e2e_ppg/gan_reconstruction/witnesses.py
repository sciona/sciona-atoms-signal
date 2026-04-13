from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_generatereconstructedppg(
    ppg_clean: AbstractSignal,
    noise: AbstractSignal,
    sampling_rate: AbstractScalar,
    generator: AbstractSignal,
    device: AbstractScalar,
) -> AbstractSignal:
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
    return AbstractArray(shape=(1,) + ppg_clean.shape, dtype="float64")
