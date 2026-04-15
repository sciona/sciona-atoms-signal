from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import witness_wrapperpredictionsignalcomputation, witness_signalarraynormalization
from kazemi_peak_detection import normalize


@register_atom(witness_wrapperpredictionsignalcomputation)
@icontract.require(lambda prediction: isinstance(prediction, np.ndarray), "prediction must be np.ndarray")
@icontract.require(lambda raw_signal: isinstance(raw_signal, np.ndarray), "raw_signal must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def wrapperpredictionsignalcomputation(prediction: np.ndarray, raw_signal: np.ndarray) -> np.ndarray:
    """Consume prediction and raw signal and return a deterministic result with no persistent state.

    Args:
        prediction: Prediction array from upstream model.
        raw_signal: Raw input signal array.

    Returns:
        Processed signal array combining prediction and raw signal.
    """
    # Combine prediction with raw signal: multiply prediction confidence by raw signal
    return prediction * raw_signal


@register_atom(witness_signalarraynormalization)
@icontract.require(lambda arr: isinstance(arr, np.ndarray), "arr must be np.ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
def signalarraynormalization(arr: np.ndarray) -> np.ndarray:
    """Normalize an input numeric array to a standard scale.

    Args:
        arr: Input array to normalize.

    Returns:
        Normalized array with same shape as input.
    """
    return normalize(arr)
