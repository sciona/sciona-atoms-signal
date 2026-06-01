"""Ghost witnesses for the FFT spectral transform atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_apply_spectral_window(
    signal: AbstractSignal | AbstractArray,
    window_type: AbstractScalar,
) -> AbstractSignal | AbstractArray:
    """Witness for apply_spectral_window."""
    _ = window_type
    if isinstance(signal, AbstractSignal):
        return AbstractSignal(
            shape=signal.shape,
            dtype=signal.dtype,
            sampling_rate=signal.sampling_rate,
            domain=signal.domain,
            units=getattr(signal, "units", "volts"),
            dim=getattr(signal, "dim", None),
        )
    return AbstractArray(
        shape=signal.shape,
        dtype=signal.dtype,
        dim=getattr(signal, "dim", None),
    )


def witness_optimize_fft_length(
    target_len: AbstractScalar,
) -> AbstractScalar:
    """Witness for optimize_fft_length."""
    return AbstractScalar(
        dtype="int64",
        min_val=target_len.min_val if target_len.min_val is not None else 1,
    )


def witness_compute_forward_rfft(
    signal: AbstractSignal | AbstractArray,
    n: AbstractScalar,
) -> AbstractSignal | AbstractArray:
    """Witness for compute_forward_rfft."""
    # Compute output length n // 2 + 1 if we have a concrete min_val, otherwise use signal shape.
    input_len = n.min_val if n.min_val is not None else (signal.shape[0] if signal.shape else 1)
    input_len = int(input_len)
    out_len = input_len // 2 + 1

    if isinstance(signal, AbstractSignal):
        return AbstractSignal(
            shape=(out_len,),
            dtype="complex128",
            sampling_rate=signal.sampling_rate,
            domain="freq",
            units=getattr(signal, "units", "volts"),
            dim=getattr(signal, "dim", None),
        )
    return AbstractArray(
        shape=(out_len,),
        dtype="complex128",
        dim=getattr(signal, "dim", None),
    )
