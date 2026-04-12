from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

from .witnesses import witness_normalizesignal, witness_wrapperevaluate


def _normalize(arr: np.ndarray) -> np.ndarray:
    minimum = float(arr.min())
    maximum = float(arr.max())
    span = maximum - minimum
    if span == 0.0:
        return np.zeros_like(arr, dtype=float)
    return (arr - minimum) / span


@register_atom(witness_normalizesignal)
@icontract.require(lambda arr: isinstance(arr, np.ndarray), "arr must be a numpy array")
@icontract.require(lambda arr: arr.size > 0, "arr must be non-empty")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "normalizesignal must return a numpy array")
def normalizesignal(arr: np.ndarray) -> np.ndarray:
    """Normalize an array into the unit interval used by the vendored Kazemi peak wrapper."""
    return _normalize(arr.astype(float, copy=False))


@register_atom(witness_wrapperevaluate)
@icontract.require(lambda prediction: isinstance(prediction, np.ndarray), "prediction must be a numpy array")
@icontract.require(lambda raw_signal: isinstance(raw_signal, np.ndarray), "raw_signal must be a numpy array")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "wrapperevaluate must return a numpy array")
def wrapperevaluate(prediction: np.ndarray, raw_signal: np.ndarray) -> np.ndarray:
    """Post-process normalized predictions against the raw signal to extract final peak indices."""
    test = _normalize(prediction.astype(float, copy=False))
    raw = raw_signal.astype(float, copy=False)

    j = 0
    indices: list[int] = []
    while j < len(test) - 3:
        if test[j] >= 0.70:
            if j < len(test) - 15:
                period = test[j : j + 15]
                period_x = raw[j : j + 15]
                index = np.asarray(np.where(period == np.max(period)))
                if len(index[0]) > 1:
                    tied = index[0].tolist()
                    max_index = np.asarray(np.where(period_x == np.max(period_x[tied])))
                    indices.append(int(max_index[0][0] + j))
                else:
                    indices.append(int(index[0][0] + j))
                j += 15
            else:
                period = test[j : j + 7]
                period_x = raw[j : j + 7]
                index = np.asarray(np.where(period == np.max(period)))
                if len(index[0]) > 1:
                    tied = index[0].tolist()
                    max_index = np.asarray(np.where(period_x == np.max(period_x[tied])))
                    indices.append(int(max_index[0][0] + j))
                else:
                    indices.append(int(index[0][0] + j))
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

    return np.array([idx for idx in indices if raw[idx] > 0], dtype=np.intp)
