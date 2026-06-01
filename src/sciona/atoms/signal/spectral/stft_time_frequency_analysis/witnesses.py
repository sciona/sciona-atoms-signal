from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_segment_into_overlapping_frames(signal: AbstractArray, frame_length: AbstractScalar | int, hop_length: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for segment_into_overlapping_frames."""
    _ = (signal, frame_length, hop_length)
    return AbstractArray(shape=signal.shape, dtype=signal.dtype)

def witness_compute_frame_fft(frames: AbstractArray, window_type: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for compute_frame_fft."""
    _ = (frames, window_type)
    return AbstractArray(shape=frames.shape, dtype=frames.dtype)

