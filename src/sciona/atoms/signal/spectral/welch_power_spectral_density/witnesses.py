from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_frame_periodograms(stft_matrix: AbstractArray, window_norm: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for compute_frame_periodograms."""
    _ = (stft_matrix, window_norm)
    return AbstractArray(shape=stft_matrix.shape, dtype=stft_matrix.dtype)

def witness_average_periodograms(periodograms: AbstractArray) -> AbstractArray:
    """Ghost witness for average_periodograms."""
    _ = (periodograms)
    return AbstractArray(shape=periodograms.shape, dtype=periodograms.dtype)

