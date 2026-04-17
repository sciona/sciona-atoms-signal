from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_zhao2018hrvanalysis(
    ecg_cleaned: AbstractSignal,
    rpeaks: AbstractArray | None,
    sampling_rate: AbstractScalar,
    window: AbstractScalar,
    mode: AbstractScalar,
) -> AbstractScalar:
    """Shape-and-type check for the Zhao 2018 ECG quality label wrapper."""
    return AbstractScalar(dtype="str")


def witness_averageqrstemplate(
    ecg_cleaned: AbstractSignal,
    rpeaks: AbstractArray | None,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for the average-QRS quality trace wrapper."""
    return AbstractSignal(
        shape=ecg_cleaned.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_cleaned, "sampling_rate", 1000.0),
        domain="time",
    )
