from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_calculatecompositesqi_zz2018(signal: AbstractSignal,
    detector_1: AbstractSignal,
    detector_2: AbstractSignal,
    fs: AbstractScalar,
    search_window: AbstractScalar,
    nseg: AbstractScalar,
    mode: AbstractScalar,
) -> AbstractArray:
    """Shape-and-type check for calculate composite sqi zz2018. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=signal.shape,
        dtype="float64",
    )
    return result

def witness_calculatebeatagreementsqi(detector_1: AbstractSignal, detector_2: AbstractSignal, fs: AbstractSignal, mode: AbstractSignal, search_window: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for calculate beat agreement sqi. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=detector_1.shape,
        dtype="float64",
        sampling_rate=getattr(detector_1, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

def witness_calculatefrequencypowersqi(ecg_signal: AbstractSignal, fs: AbstractSignal, nseg: AbstractSignal, num_spectrum: AbstractSignal, dem_spectrum: AbstractSignal, mode: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for calculate frequency power sqi. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=ecg_signal.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result

def witness_calculatekurtosissqi(signal: AbstractSignal, fisher: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for calculate kurtosis sqi. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, 'sampling_rate_prime', 44100.0),
        domain="time",
    )
    return result
