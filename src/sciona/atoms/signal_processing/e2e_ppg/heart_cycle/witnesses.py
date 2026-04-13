from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_detect_heart_cycles(
    ppg: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    return AbstractSignal(
        shape=ppg.shape,
        dtype="float64",
        sampling_rate=getattr(ppg, "sampling_rate", 44100.0),
        domain="time",
    )


def witness_heart_cycle_detection(
    ppg: AbstractArray,
    sampling_rate: AbstractScalar,
) -> AbstractArray:
    return AbstractArray(shape=ppg.shape, dtype="float64")
