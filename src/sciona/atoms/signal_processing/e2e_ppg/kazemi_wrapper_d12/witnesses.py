from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractSignal


def witness_normalizesignal(arr: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for normalize signal."""
    return AbstractSignal(
        shape=arr.shape,
        dtype="float64",
        sampling_rate=getattr(arr, "sampling_rate", 44100.0),
        domain="time",
    )


def witness_wrapperevaluate(prediction: AbstractArray, raw_signal: AbstractArray) -> AbstractArray:
    """Shape-and-type check for wrapper evaluate."""
    return AbstractArray(shape=(0,), dtype="int64")
