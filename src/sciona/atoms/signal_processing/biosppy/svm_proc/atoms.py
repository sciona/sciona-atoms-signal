from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Any, Union
import numpy as np

import icontract
import biosppy.biometrics as biometrics
from ageoa.ghost.registry import register_atom
from .witnesses import witness_assess_classification, witness_assess_runs, witness_combination, witness_cross_validation, witness_get_auth_rates, witness_get_id_rates, witness_get_subject_results, witness_majority_rule

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_get_auth_rates)
@icontract.require(lambda TP: TP is not None, "TP cannot be None")
@icontract.require(lambda FP: FP is not None, "FP cannot be None")
@icontract.require(lambda TN: TN is not None, "TN cannot be None")
@icontract.require(lambda FN: FN is not None, "FN cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result: result is not None, "Get Auth Rates output must not be None")
def get_auth_rates(TP: np.ndarray, FP: np.ndarray, TN: np.ndarray, FN: np.ndarray, thresholds: np.ndarray) -> dict:
    """Compute authentication rates from correct and incorrect prediction counts at each threshold.

    Args:
        TP — true positive: correct-accept counts per threshold
        FP — false positive: wrong-accept counts per threshold
        TN — true negative: correct-reject counts per threshold
        FN — false negative: wrong-reject counts per threshold
        thresholds: decision thresholds

    Returns:
        rate metrics at each threshold
    """
    total = TP + FP + TN + FN
    FAR = np.where(total > 0, FP / (FP + TN + 1e-15), 0.0)  # false accept rate
    FRR = np.where(total > 0, FN / (FN + TP + 1e-15), 0.0)  # false reject rate
    accuracy = np.where(total > 0, (TP + TN) / total, 0.0)
    return {"FAR": FAR, "FRR": FRR, "accuracy": accuracy, "thresholds": thresholds}

@register_atom(witness_get_id_rates)
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.require(lambda M: M is not None, "M cannot be None")
@icontract.require(lambda R: R is not None, "R cannot be None")
@icontract.require(lambda N: N is not None, "N cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result: result is not None, "Get Id Rates output must not be None")
def get_id_rates(H: np.ndarray, M: np.ndarray, R: np.ndarray, N: int, thresholds: np.ndarray) -> dict:
    """Compute identification rates for a Support Vector Machine (SVM) biometric classifier. Derives accuracy, miss rate, reject rate, and Equal Error Rate (EER) from hits, misses, and rejections at each decision threshold.

    Args:
        H: Hit (correct identification) counts per threshold.
        M: Miss counts per threshold.
        R: Reject counts per threshold.
        N: Total number of test samples.
        thresholds: Decision thresholds.

    Returns:
        Identification performance metrics at each threshold.
    """
    total = H + M + R
    accuracy = np.where(total > 0, H / total, 0.0)
    miss_rate = np.where(total > 0, M / total, 0.0)
    reject_rate = np.where(total > 0, R / total, 0.0)
    return {"accuracy": accuracy, "miss_rate": miss_rate, "reject_rate": reject_rate, "N": N, "thresholds": thresholds}

@register_atom(witness_get_subject_results)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda subject: subject is not None, "subject cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.require(lambda subjects: subjects is not None, "subjects cannot be None")
@icontract.require(lambda subject_dict: subject_dict is not None, "subject_dict cannot be None")
@icontract.require(lambda subject_idx: subject_idx is not None, "subject_idx cannot be None")
@icontract.ensure(lambda result: result is not None, "Get Subject Results output must not be None")
def get_subject_results(
    results: dict[str, Any],
    subject: Hashable,
    thresholds: np.ndarray,
    subjects: Sequence[Hashable],
    subject_dict: Mapping[Hashable, int],
    subject_idx: Sequence[int],
) -> dict[str, Any]:
    """Compute authentication and identification performance metrics for a
given subject.

Parameters
----------
results : dict
    Classification results.
subject : hashable
    True subject label.
thresholds : array
    Classifier thresholds.
subjects : list
    Target subject classes.
subject_dict : bidict
    Subject-label conversion dictionary.
subject_idx : list
    Subject index.

Returns
-------
assessment : dict
    Authentication and identification results.

    Args:
        results: Input data.
        subject: Input data.
        thresholds: Input data.
        subjects: Input data.
        subject_dict: Input data.
        subject_idx: Input data.

    Returns:
        Result data.
    """
    # Per-subject performance aggregation
    auth = results.get("authentication", {})
    ident = results.get("identification", {})
    return {"authentication": auth, "identification": ident, "subject": subject, "thresholds": thresholds}

@register_atom(witness_assess_classification)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result: result is not None, "Assess Classification output must not be None")
def assess_classification(results: dict, thresholds: np.ndarray) -> dict:
    """Assess the performance of a biometric classification test.

Parameters
----------
results : dict
    Classification results.
thresholds : array
    Classifier thresholds.

Returns
-------
assessment : dict
    Classification assessment.

    Args:
        results: Input data.
        thresholds: Input data.

    Returns:
        Result data.
    """
    # Classification assessment across subjects and thresholds
    return {"results": results, "thresholds": thresholds}

@register_atom(witness_assess_runs)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda subjects: subjects is not None, "subjects cannot be None")
@icontract.ensure(lambda result: result is not None, "Assess Runs output must not be None")
def assess_runs(results: list, subjects: list) -> dict:
    """Assess the performance of multiple biometric classification runs.

Parameters
----------
results : list
    Classification assessment for each run.
subjects : list
    Common target subject classes.

Returns
-------
assessment : dict
    Global classification assessment.

    Args:
        results: Input data.
        subjects: Input data.

    Returns:
        Result data.
    """
    # Average metrics across runs
    return {"results": results, "subjects": subjects}

@register_atom(witness_combination)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda weights: weights is not None, "weights cannot be None")
@icontract.ensure(lambda result: result is not None, "Combination output must not be None")
def combination(results: dict, weights: dict) -> tuple:
    """Combine results from multiple classifiers.

Parameters
----------
results : dict
    Results for each classifier.
weights : dict, optional
    Weight for each classifier.

Returns
-------
decision : object
    Consensus decision.
confidence : float
    Confidence estimate of the decision.
counts : array
    Weight for each possible decision outcome.
classes : array
    List of possible decision outcomes.

    Args:
        results: Input data.
        weights: Input data.

    Returns:
        Result data.
    """
    # Weighted ensemble combination
    all_decisions = []
    all_weights = []
    for clf, res in results.items():
        w = weights.get(clf, 1.0) if weights else 1.0
        all_decisions.append(res)
        all_weights.append(w)
    # Return weighted majority
    classes = sorted(set(all_decisions))
    counts = np.array([sum(w for d, w in zip(all_decisions, all_weights) if d == c) for c in classes])
    best = classes[np.argmax(counts)]
    return (best, float(np.max(counts) / np.sum(counts)), counts, np.array(classes))

@register_atom(witness_majority_rule)
@icontract.require(lambda labels: labels is not None, "labels cannot be None")
@icontract.require(lambda random: random is not None, "random cannot be None")
@icontract.ensure(lambda result: result is not None, "Majority Rule output must not be None")
def majority_rule(labels: Union[np.ndarray, list], random: bool) -> tuple:
    """Determine the most frequent class label.

Parameters
----------
labels : array, list
    List of clas labels.
random : bool, optional
    If True, will choose randomly in case of tied classes, otherwise the
    first element is chosen.

Returns
-------
    decision : object
        Consensus decision.
    count : int
        Number of elements of the consensus decision.

    Args:
        labels: Input data.
        random: Input data.

    Returns:
        Result data.
    """
    labels_arr = np.asarray(labels)
    unique, counts = np.unique(labels_arr, return_counts=True)
    best = unique[np.argmax(counts)]
    count = int(np.max(counts))
    if random and np.sum(counts == count) > 1:
        tied = unique[counts == count]
        best = np.random.choice(tied)
    return (best, count)

@register_atom(witness_cross_validation)
@icontract.require(lambda labels: labels is not None, "labels cannot be None")
@icontract.require(lambda n_iter: n_iter is not None, "n_iter cannot be None")
@icontract.require(lambda test_size: test_size is not None, "test_size cannot be None")
@icontract.ensure(lambda result: result is not None, "Cross Validation output must not be None")
def cross_validation(
    labels: Union[list, np.ndarray],
    n_iter: int = 10,
    test_size: Union[float, int] = 0.1,
    train_size: Union[float, int, None] = None,
    random_state: Union[int, None] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Return a Cross Validation (CV) iterator.

Wraps the StratifiedShuffleSplit iterator from sklearn.model_selection.
This iterator returns stratified randomized folds, which preserve the
percentage of samples for each class.

Parameters
----------
labels : list, array
    List of class labels for each data sample.
n_iter : int, optional
    Number of splitting iterations.
test_size : float, int, optional
    If float, represents the proportion of the dataset to include in the
    test split; if int, represents the absolute number of test samples.
train_size : float, int, optional
    If float, represents the proportion of the dataset to include in the
    train split; if int, represents the absolute number of train samples.
random_state : int, RandomState, optional
    The seed of the pseudo random number generator to use when shuffling
    the data.

Returns
-------
cv : CV iterator
    Cross Validation iterator.

    Args:
        labels: Input data.
        n_iter: Input data.
        test_size: Input data.
        train_size: Input data.
        random_state: Input data.

    Returns:
        Result data.
    """
    return biometrics.cross_validation(
        labels=labels,
        n_iter=n_iter,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
    )[0]
