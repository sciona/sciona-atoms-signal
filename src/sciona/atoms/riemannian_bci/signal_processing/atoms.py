"""Riemannian BCI signal processing atoms.

Aggregation and ensemble utilities derived from Barachant's Kaggle competition
solutions (BSD 3-Clause).  No pyRiemann dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.stats

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_segment_max_aggregation,
    witness_ranked_prediction_blend,
)


@register_atom(witness_segment_max_aggregation)
@icontract.require(lambda predictions: predictions.ndim == 2, "predictions must be 2-D (n_windows, n_classes)")
@icontract.require(lambda window_size: window_size >= 1, "window_size must be >= 1")
@icontract.require(lambda stride: stride >= 1, "stride must be >= 1")
@icontract.require(lambda n_samples: n_samples >= 1, "n_samples must be >= 1")
@icontract.ensure(
    lambda result, n_samples, predictions: result.shape == (n_samples, predictions.shape[1]),
    "output shape must be (n_samples, n_classes)",
)
def segment_max_aggregation(
    predictions: NDArray[np.float64],
    window_size: int,
    stride: int,
    n_samples: int,
) -> NDArray[np.float64]:
    """Aggregate windowed predictions by element-wise max over overlapping segments.

    For each window *i*, its prediction vector is assigned to samples
    ``[i*stride, i*stride + window_size)``.  Where segments overlap the
    element-wise maximum is taken.

    Args:
        predictions: Prediction array of shape (n_windows, n_classes).
        window_size: Number of samples each prediction window covers.
        stride: Step size between consecutive windows.
        n_samples: Total number of output samples.

    Returns:
        Per-sample aggregated predictions of shape (n_samples, n_classes).
    """
    n_windows, n_classes = predictions.shape
    # Initialize with -inf so the first assignment via max always wins
    result = np.full((n_samples, n_classes), -np.inf, dtype=np.float64)
    covered = np.zeros(n_samples, dtype=bool)

    for i in range(n_windows):
        start = i * stride
        end = min(start + window_size, n_samples)
        if start >= n_samples:
            break
        result[start:end] = np.maximum(result[start:end], predictions[i])
        covered[start:end] = True

    # For any uncovered samples, set to zero
    result[~covered] = 0.0
    return result


@register_atom(witness_ranked_prediction_blend)
@icontract.require(
    lambda predictions: predictions.ndim == 2,
    "predictions must be 2-D (n_models, n_samples)",
)
@icontract.require(
    lambda weights: weights.ndim == 1,
    "weights must be 1-D (n_models,)",
)
@icontract.require(
    lambda predictions, weights: predictions.shape[0] == weights.shape[0],
    "number of weight entries must match number of models",
)
@icontract.ensure(
    lambda result, predictions: result.shape == (predictions.shape[1],),
    "output shape must be (n_samples,)",
)
def ranked_prediction_blend(
    predictions: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Blend predictions from multiple models by weighted rank-averaging.

    Each model's predictions are converted to ranks, combined as a weighted
    average, and the result is re-ranked to produce the final blended output.

    Args:
        predictions: Array of shape (n_models, n_samples).
        weights: Array of shape (n_models,) with per-model blend weights.

    Returns:
        Blended prediction of shape (n_samples,).
    """
    n_models, n_samples = predictions.shape
    w = weights / weights.sum()

    ranked = np.zeros((n_models, n_samples), dtype=np.float64)
    for i in range(n_models):
        ranked[i] = scipy.stats.rankdata(predictions[i])

    blended = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_models):
        blended += w[i] * ranked[i]

    # Re-rank the blended result
    return scipy.stats.rankdata(blended).astype(np.float64)
