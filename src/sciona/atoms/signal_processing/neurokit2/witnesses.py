from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_zhao2018hrvanalysis(
    ecg_cleaned: AbstractSignal,
    rpeaks: AbstractArray | None,
    sampling_rate: AbstractScalar,
    window: AbstractScalar,
    mode: AbstractScalar,
) -> AbstractScalar:
    """Shape-and-type check for zhao2018 hrv analysis."""
    return AbstractScalar(dtype="str")


def witness_averageqrstemplate(
    ecg_cleaned: AbstractSignal,
    rpeaks: AbstractArray | None,
    sampling_rate: AbstractScalar,
) -> AbstractSignal:
    """Shape-and-type check for average qrs template."""
    return AbstractSignal(
        shape=ecg_cleaned.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_cleaned, "sampling_rate", 1000.0),
        domain="time",
    )
