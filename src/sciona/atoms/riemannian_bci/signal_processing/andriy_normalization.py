"""Joint train/valid-prediction normalization defined by the Andriy source."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.registry import register_atom


def witness_andriy_joint_normalization(training_features, prediction_features, valid_prediction_rows):
    return training_features, prediction_features


@register_atom(witness_andriy_joint_normalization)
def andriy_joint_normalization(training_features: NDArray[np.float64], prediction_features: NDArray[np.float64],
                                valid_prediction_rows: NDArray[np.bool_]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize row/feature matrices using training plus valid prediction rows.

    Source normalise_tr_mv fits feature means and sample standard deviations
    (N-1 denominator) to concatenated training and valid prediction observations.
    normalise_te_mv applies the same values to both complete partitions. This
    is intentionally transductive: valid prediction rows influence training.
    Boolean mask is the explicit zero-based equivalent of source goodidx.
    Training and selected prediction rows must be finite; unselected prediction
    rows may contain NaN but not infinity. At least two fit rows are required.
    Zero-variance features produce source NaN/inf and remain for later model
    handling; no epsilon or replacement standard deviation is introduced.
    Outputs preserve row/feature order, are independent float64 arrays, and do
    not modify inputs. Formula reconstruction, not MATLAB runtime validation.
    """
    train, prediction, valid = np.asarray(training_features), np.asarray(prediction_features), np.asarray(valid_prediction_rows)
    for value in [train, prediction]:
        if value.dtype.kind not in 'iuf' or value.ndim != 2 or min(value.shape) == 0 or np.any(np.isinf(value)):
            raise ValueError('nonempty real row/feature matrices without infinities required')
    if train.shape[1] != prediction.shape[1] or valid.dtype.kind != 'b' or valid.shape != (len(prediction),):
        raise ValueError('matching feature widths and boolean prediction-row mask required')
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(prediction[valid])):
        raise ValueError('normalization fit rows must be finite')
    fit = np.concatenate([train, prediction[valid]]).astype(float)
    if len(fit) < 2:
        raise ValueError('at least two normalization fit rows required')
    mean, deviation = fit.mean(axis=0), fit.std(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (train.astype(float)-mean)/deviation, (prediction.astype(float)-mean)/deviation
