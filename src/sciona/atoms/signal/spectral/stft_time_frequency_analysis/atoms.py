from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_segment_into_overlapping_frames,
    witness_compute_frame_fft,
)

@register_atom(witness_segment_into_overlapping_frames, name="segment_into_overlapping_frames")
@icontract.require(lambda signal, frame_length, hop_length: signal.ndim == 1, "Precondition failed: signal.ndim == 1")
@icontract.require(lambda signal, frame_length, hop_length: frame_length > 0, "Precondition failed: frame_length > 0")
@icontract.require(lambda signal, frame_length, hop_length: hop_length > 0, "Precondition failed: hop_length > 0")
@icontract.require(lambda signal, frame_length, hop_length: hop_length <= frame_length, "Precondition failed: hop_length <= frame_length")
@icontract.ensure(lambda result, signal, frame_length, hop_length: result.ndim == 2, "Postcondition failed: result.ndim == 2")
@icontract.ensure(lambda result, signal, frame_length, hop_length: result.shape[0] == frame_length, "Postcondition failed: result.shape[0] == frame_length")
def segment_into_overlapping_frames(signal: NDArray[np.float64], frame_length: int, hop_length: int) -> NDArray[np.float64]:
    """Segments a 1D signal into overlapping frames with specified window size and hop length.

    Args:
        signal: NDArray[np.float64]
        frame_length: int
        hop_length: int

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.signal
    return scipy.signal.stft(signal=signal, frame_length=frame_length, hop_length=hop_length) # type: ignore

@register_atom(witness_compute_frame_fft, name="compute_frame_fft")
@icontract.require(lambda frames, window_type: frames.ndim == 2, "Precondition failed: frames.ndim == 2")
@icontract.ensure(lambda result, frames, window_type: result.ndim == 2, "Postcondition failed: result.ndim == 2")
@icontract.ensure(lambda result, frames, window_type: result.shape[0] == frames.shape[0] // 2 + 1, "Postcondition failed: result.shape[0] == frames.shape[0] // 2 + 1")
def compute_frame_fft(frames: NDArray[np.float64], window_type: str) -> NDArray[np.complex128]:
    """Applies a window and computes the DFT / Fast Fourier Transform (FFT) for each signal frame.

    This corresponds to the Fourier Transform / Fourier Transform Windowed / compute_fft step of spectral analysis.

    Args:
        frames: 2D array of shape (frame_length, num_frames)
        window_type: str

    Returns:
        result: NDArray[np.complex128]
    """
    import scipy.fft
    return scipy.fft.rfft(frames=frames, window_type=window_type) # type: ignore

