from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from bidict import bidict
import biosppy.biometrics as biometrics

from sciona.atoms.signal_processing.biosppy.svm_proc import (
    assess_classification,
    assess_runs,
    combination,
    cross_validation,
    get_auth_rates,
    get_id_rates,
    get_subject_results,
    majority_rule,
)
from sciona.probes.signal_processing.biosppy_svm_proc import probe_records


def _assert_equivalent(actual: object, expected: object) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        if expected.dtype == object or actual.dtype == object:
            assert actual.tolist() == expected.tolist()
        else:
            np.testing.assert_allclose(actual, expected, equal_nan=True)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_equivalent(actual[key], expected[key])
    elif isinstance(expected, Sequence) and not isinstance(expected, str):
        assert isinstance(actual, Sequence)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _assert_equivalent(got, want)
    elif isinstance(expected, np.generic):
        assert actual == expected.item()
    else:
        assert actual == expected


def _subject_inputs() -> tuple[
    dict[str, object],
    np.ndarray,
    list[str],
    bidict[str, int],
    list[int],
]:
    thresholds = np.array([0.25, 0.75], dtype=float)
    subjects = ["alice", "bob"]
    subject_dict: bidict[str, int] = bidict({"alice": 0, "bob": 1})
    subject_idx = [0, 1]
    results = {
        "authentication": np.array(
            [
                [[True, True, False, True], [False, True, False, False]],
                [[True, False, False, True], [False, False, False, False]],
            ],
            dtype=bool,
        ),
        "identification": np.array(
            [
                [0, 1, "", 0],
                [0, "", "", 0],
            ],
            dtype=object,
        ),
    }
    return results, thresholds, subjects, subject_dict, subject_idx


def _classification_inputs() -> tuple[dict[str, object], np.ndarray]:
    alice_results, thresholds, subjects, subject_dict, subject_idx = _subject_inputs()
    bob_results = {
        "authentication": np.array(
            [
                [[False, False, True], [True, True, False]],
                [[False, False, False], [True, False, False]],
            ],
            dtype=bool,
        ),
        "identification": np.array(
            [
                [1, 0, ""],
                [1, "", ""],
            ],
            dtype=object,
        ),
    }
    results: dict[str, object] = {
        "subjectList": subjects,
        "subjectDict": subject_dict,
        "alice": alice_results,
        "bob": bob_results,
    }
    assert subject_idx == [0, 1]
    return results, thresholds


def test_biosppy_svm_proc_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.svm_proc")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_svm_proc")
    assert hasattr(atoms, "get_auth_rates")
    assert hasattr(atoms, "majority_rule")
    assert hasattr(probes, "SVM_PROC_PROBE_TARGETS")


def test_probe_records_resolve_to_live_symbols() -> None:
    for record in probe_records():
        module = importlib.import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))


def test_get_auth_rates_matches_upstream_metrics() -> None:
    thresholds = np.array([0.1, 0.4, 0.9], dtype=float)
    TP = np.array([9.0, 7.0, 4.0])
    FP = np.array([3.0, 2.0, 1.0])
    TN = np.array([5.0, 6.0, 8.0])
    FN = np.array([1.0, 3.0, 6.0])

    actual = get_auth_rates(TP=TP, FP=FP, TN=TN, FN=FN, thresholds=thresholds)
    expected = biometrics.get_auth_rates(
        TP=TP,
        FP=FP,
        TN=TN,
        FN=FN,
        thresholds=thresholds,
    ).as_dict()

    _assert_equivalent(actual, expected)
    assert set(actual) == {"Acc", "TAR", "FAR", "FRR", "TRR", "EER", "Err", "PPV", "FDR", "NPV", "FOR", "MCC"}


def test_get_id_rates_matches_upstream_normalization_by_n() -> None:
    thresholds = np.array([0.0, 0.5, 1.0], dtype=float)
    H = np.array([8.0, 6.0, 3.0])
    M = np.array([1.0, 3.0, 2.0])
    R = np.array([0.0, 1.0, 5.0])
    N = 12

    actual = get_id_rates(H=H, M=M, R=R, N=N, thresholds=thresholds)
    expected = biometrics.get_id_rates(H=H, M=M, R=R, N=N, thresholds=thresholds).as_dict()

    _assert_equivalent(actual, expected)
    np.testing.assert_allclose(actual["Acc"], H / N)


def test_get_subject_results_matches_upstream_subject_assessment() -> None:
    results, thresholds, subjects, subject_dict, subject_idx = _subject_inputs()

    actual = get_subject_results(
        results=results,
        subject="alice",
        thresholds=thresholds,
        subjects=subjects,
        subject_dict=subject_dict,
        subject_idx=subject_idx,
    )
    expected = biometrics.get_subject_results(
        results=results,
        subject="alice",
        thresholds=thresholds,
        subjects=subjects,
        subject_dict=subject_dict,
        subject_idx=subject_idx,
    )[0]

    _assert_equivalent(actual, expected)


def test_assess_classification_matches_upstream_global_and_subject_assessment() -> None:
    results, thresholds = _classification_inputs()

    actual = assess_classification(results=results, thresholds=thresholds)
    expected = biometrics.assess_classification(results=results, thresholds=thresholds)[0]

    _assert_equivalent(actual, expected)


def test_assess_runs_matches_upstream_multi_run_assessment() -> None:
    results, thresholds = _classification_inputs()
    run = assess_classification(results=results, thresholds=thresholds)

    actual = assess_runs(results=[run, run], subjects=["alice", "bob"])
    expected = biometrics.assess_runs(results=[run, run], subjects=["alice", "bob"])[0]

    _assert_equivalent(actual, expected)


def test_combination_matches_upstream_for_list_valued_classifier_outputs() -> None:
    results = {
        "clf_a": ["alice", "alice", "bob"],
        "clf_b": ["bob", "alice"],
        "clf_c": np.array(["alice", "bob", "bob"], dtype=object),
    }
    weights = {"clf_a": 1.0, "clf_b": 0.5, "clf_c": 2.0}

    actual = combination(results=results, weights=weights)
    expected = tuple(biometrics.combination(results=results, weights=weights))

    _assert_equivalent(actual, expected)


def test_combination_raises_upstream_error_for_empty_results() -> None:
    with pytest.raises(biometrics.CombinationError):
        combination(results={})


def test_majority_rule_matches_upstream_without_random_tie_breaking() -> None:
    labels = np.array(["bob", "alice", "alice", "bob", "alice"], dtype=object)

    actual = majority_rule(labels=labels, random=False)
    expected = tuple(biometrics.majority_rule(labels=labels, random=False))

    _assert_equivalent(actual, expected)


def test_cross_validation_matches_upstream_first_split() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0], dtype=int)

    wrapped_split = next(cross_validation(labels, n_iter=2, test_size=0.33, train_size=None, random_state=7))
    upstream_split = next(
        biometrics.cross_validation(
            labels=labels,
            n_iter=2,
            test_size=0.33,
            train_size=None,
            random_state=7,
        )[0]
    )

    assert np.array_equal(wrapped_split[0], upstream_split[0])
    assert np.array_equal(wrapped_split[1], upstream_split[1])
