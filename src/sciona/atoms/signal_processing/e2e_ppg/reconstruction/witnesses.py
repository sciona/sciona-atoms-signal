from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_gan_patch_reconstruction(ppg_clean: AbstractSignal, noise: AbstractSignal, sampling_rate: AbstractScalar, generator: AbstractSignal, device: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for gan patch reconstruction. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=ppg_clean.shape,
        dtype="float64",
        sampling_rate=getattr(ppg_clean, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result

def witness_windowed_signal_reconstruction(sig: AbstractSignal, clean_indices: AbstractSignal, noisy_indices: AbstractSignal, sampling_rate: AbstractSignal, filter_signal: AbstractSignal) -> AbstractSignal:
    """Shape-and-type check for windowed signal reconstruction. Returns output metadata without running the real computation."""
    result = AbstractSignal(
        shape=sig.shape,
        dtype="float64",
        sampling_rate=getattr(sig, 'sampling_rate', 44100.0),
        domain="time",
    )
    return result
