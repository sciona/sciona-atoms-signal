from __future__ import annotations
"""Kazemi peak-detection post-processing atoms."""


import numpy as np
import icontract
from sciona.ghost.registry import register_atom

from .._vendor import load_e2e_ppg_module
from .witnesses import witness_wrapperpredictionsignalcomputation, witness_signalarraynormalization


def _as_vector(values: np.ndarray) -> np.ndarray:
    """Return a float vector while preserving the upstream row-wise scan contract."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def _normalize_prediction(prediction: np.ndarray) -> np.ndarray:
    arr = _as_vector(prediction)
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    span = maximum - minimum
    if span == 0.0:
        return np.zeros_like(arr, dtype=float)
    return (arr - minimum) / span


@register_atom(witness_wrapperpredictionsignalcomputation)
@icontract.require(lambda prediction: isinstance(prediction, np.ndarray), "prediction must be np.ndarray")
@icontract.require(lambda raw_signal: isinstance(raw_signal, np.ndarray), "raw_signal must be np.ndarray")
@icontract.require(lambda prediction: prediction.size > 0, "prediction must be non-empty")
@icontract.require(lambda raw_signal: raw_signal.size > 0, "raw_signal must be non-empty")
@icontract.require(
    lambda prediction, raw_signal: np.asarray(prediction).reshape(-1).shape[0]
    == np.asarray(raw_signal).reshape(-1).shape[0],
    "prediction and raw_signal must have the same flattened length",
)
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be np.ndarray")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "result must contain integer indices")
def wrapperpredictionsignalcomputation(prediction: np.ndarray, raw_signal: np.ndarray) -> np.ndarray:
    """Extract Kazemi wrapper peak indices from model predictions and a raw PPG segment.

    Args:
        prediction: Model score trace for one PPG segment.
        raw_signal: Raw PPG trace aligned to the prediction trace.

    Returns:
        Integer indices for retained positive-amplitude peaks.
    """
    scores = _normalize_prediction(prediction)
    raw = _as_vector(raw_signal)

    j = 0
    indices: list[int] = []
    while j < len(scores) - 3:
        if scores[j] >= 0.70:
            if j < len(scores) - 15:
                period = scores[j : j + 15]
                period_raw = raw[j : j + 15]
                tied = np.flatnonzero(period == np.max(period))
                if len(tied) > 1:
                    local_index = int(tied[np.argmax(period_raw[tied])])
                else:
                    local_index = int(tied[0])
                indices.append(local_index + j)
                j += 15
            else:
                period = scores[j : j + 7]
                period_raw = raw[j : j + 7]
                tied = np.flatnonzero(period == np.max(period))
                if len(tied) > 1:
                    local_index = int(tied[np.argmax(period_raw[tied])])
                    indices.append(local_index + j)
                j += 7
        else:
            j += 1

    e = 0
    while e < len(indices) - 1:
        if indices[e + 1] - indices[e] < 35:
            if raw[indices[e + 1]] < raw[indices[e]]:
                del indices[e + 1]
            else:
                del indices[e]
        else:
            e += 1

    return np.array([idx for idx in indices if raw[idx] > 0.0], dtype=np.intp)


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
    module = load_e2e_ppg_module("kazemi_peak_detection")
    return module.normalize(arr)
