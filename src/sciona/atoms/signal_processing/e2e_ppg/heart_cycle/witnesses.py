from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_detect_heart_cycles(
    ppg: AbstractSignal,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Describe heart-cycle boundary output as a time-domain signal."""
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
    """Describe heart-cycle boundary output as an array of detections."""
    return AbstractArray(shape=ppg.shape, dtype="float64")
