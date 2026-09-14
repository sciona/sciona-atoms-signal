# SPDX-License-Identifier: GPL-2.0-or-later
"""Raw-clip Andriy preprocessing, CSP and normalized population inputs."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.registry import register_atom
from .andriy_preprocessing import andriy_documented_preprocess_clip,andriy_csp_training_classes
from .andriy_csp import andriy_csp_filters,andriy_csp_project
from .andriy_clip_features import andriy_documented_clip_features
from .andriy_population_inputs import andriy_population_inputs,witness_andriy_population_inputs


def witness_andriy_documented_population(csp_candidates: list,candidate_sequences,primary_positive: list,
                                        auxiliary_positive: list,negative: list,prediction: list,sampling_frequency: int):
    return witness_andriy_population_inputs(primary_positive,auxiliary_positive,negative,prediction)


@register_atom(witness_andriy_documented_population)
def andriy_documented_population(csp_candidates: list,candidate_sequences: NDArray[np.int64],primary_positive: list,
                                 auxiliary_positive: list,negative: list,prediction: list,sampling_frequency: int) -> tuple[NDArray[np.float64],NDArray[np.int64],NDArray[np.float64],NDArray[np.bool_]]:
    """Produce one population's model-ready arrays from ordered raw source clips.

    Each clip is finite 16-channel/exactly600-second samples at a common integer
    rate >120Hz. CSP candidates represent the complete ordered positive-file
    enumeration, including the first two entries skipped by source selection.
    Corresponding sequence positions are 1..6. Training/prediction group lists
    are caller-supplied source-safe memberships in original order; no membership,
    labels, filenames or ordering is inferred. Primary/auxiliary positive groups
    may individually be empty; their union, negatives and predictions cannot.

    Preprocess CSP candidates to128Hz, select sequence6/1 classes and fit once.
    For each group clip, preprocess to256/128Hz, project its first CSP component,
    extract the complete1965-column feature matrix and apply source population
    grouping/joint normalization. Return train, 1/0 labels, prediction and mask.
    All helper runtime limits remain: documented resampling/Welch/AR initial
    state, current SciPy connectivity, canonical CSP signs and fail-closed
    degenerate covariance checks. No historical MATLAB or model accuracy claim.
    No persistence, input mutation, model training or cross-population pooling.
    """
    groups=[primary_positive,auxiliary_positive,negative,prediction]
    seq=np.asarray(candidate_sequences)
    if (not isinstance(csp_candidates,list) or len(csp_candidates)<4 or
        seq.dtype.kind not in 'iu' or seq.shape!=(len(csp_candidates),) or np.any((seq<1)|(seq>6))):
        raise ValueError('ordered CSP candidates and matching sequence positions 1..6 required')
    if any(not isinstance(group,list) for group in groups) or not (primary_positive or auxiliary_positive) or not negative or not prediction:
        raise ValueError('ordered source-safe groups must supply both classes and prediction clips')
    if isinstance(sampling_frequency,(bool,np.bool_)) or not isinstance(sampling_frequency,(int,np.integer)) or sampling_frequency<=120:
        raise ValueError('common integer source sampling frequency above 120 Hz required')
    for group in [csp_candidates,*groups]:
        for value in group:
            x=np.asarray(value)
            if x.dtype.kind not in 'iuf' or x.shape!=(16,600*sampling_frequency) or not np.all(np.isfinite(x)):
                raise ValueError('all raw clips must be finite 16-channel/exactly600-second matrices')
    low_candidates=[]
    for clip in csp_candidates:
        _,low=andriy_documented_preprocess_clip(clip,sampling_frequency)
        low_candidates.append(low)
    late,early=andriy_csp_training_classes(low_candidates,seq)
    filters=andriy_csp_filters(late,early)
    del low_candidates,late,early
    feature_groups=[]
    for group in groups:
        features=[]
        for clip in group:
            high,low=andriy_documented_preprocess_clip(clip,sampling_frequency)
            projected=andriy_csp_project(low,filters)
            matrix,_=andriy_documented_clip_features(high,projected)
            features.append(matrix)
        feature_groups.append(features)
    return andriy_population_inputs(*feature_groups)
