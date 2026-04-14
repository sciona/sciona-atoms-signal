from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_thresholdbasedsignalsegmentation(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    Pth: AbstractScalar,
) -> AbstractSignal:
    """Describe the ASI threshold-based detector output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_asi_signal_segmenter(
    signal: AbstractSignal,
    sampling_rate: AbstractSignal,
    Pth: AbstractSignal,
) -> AbstractSignal:
    """Describe the ASI segmenter output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_christovqrsdetect(signal: AbstractSignal, sampling_rate: AbstractScalar) -> AbstractArray:
    """Describe Christov detector output as integer peak indices."""
    return AbstractArray(shape=(signal.shape[0],), dtype="int64", min_val=0, max_val=signal.shape[0] - 1)


def witness_christov_qrs_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    """Describe the alias Christov segmenter output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_engzee_signal_segmentation(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    threshold: AbstractScalar,
) -> AbstractArray:
    """Describe Engzee detector output as integer peak indices."""
    return AbstractArray(shape=(signal.shape[0],), dtype="int64", min_val=0, max_val=signal.shape[0] - 1)


def witness_engzee_qrs_segmentation(signal: AbstractSignal, sampling_rate: AbstractSignal, threshold: AbstractSignal) -> AbstractSignal:
    """Describe the alias Engzee segmenter output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_gamboa_segmentation(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    tol: AbstractScalar,
) -> AbstractSignal:
    """Describe Gamboa detector output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_gamboa_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal, tol: AbstractSignal) -> AbstractSignal:
    """Describe the alias Gamboa segmenter output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_hamilton_segmentation(signal: AbstractSignal, sampling_rate: AbstractScalar) -> AbstractSignal:
    """Describe Hamilton detector output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_hamilton_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    """Describe the alias Hamilton segmenter output as a time-domain signal."""
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )
