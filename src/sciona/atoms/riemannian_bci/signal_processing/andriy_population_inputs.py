"""Source training-group assembly and joint population normalization."""
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sciona.ghost.abstract import AbstractArray,AbstractMatrix
from sciona.ghost.registry import register_atom
from .andriy_normalization import andriy_joint_normalization


class _AbstractRetainedLabels(BaseModel):
    """A vector whose length depends on the numerical training-row mask."""
    shape: tuple[str] = ('retained_training_windows',)
    dtype: str = 'int64'


def witness_andriy_population_inputs(primary_positive: list,auxiliary_positive: list,negative: list,prediction: list):
    count=19*len(prediction)
    return (AbstractMatrix(shape=('retained_training_windows','1965'),dtype='float64'),
            _AbstractRetainedLabels(),AbstractArray(shape=(count,1965),dtype='float64'),
            AbstractArray(shape=(count,),dtype='bool'))


@register_atom(witness_andriy_population_inputs)
def andriy_population_inputs(primary_positive: list,auxiliary_positive: list,negative: list,prediction: list) -> tuple[NDArray[np.float64],NDArray[np.int64],NDArray[np.float64],NDArray[np.bool_]]:
    """Prepare one population's normalized model inputs from ordered clip features.

    Each list entry is an already imputed/transformed 19-by-1965 clip matrix.
    Lists correspond to source-safe P, auxiliary labelled-positive P1, I and
    prediction membership; callers must supply membership and original order.
    These are not the CSP sequence groups. Primary/auxiliary positive groups
    may individually be empty, but retained positive and negative rows must
    both exist. Discard NaN-containing training rows, preserving P/P1/I order.
    Keep all prediction rows and compute their pre-normalization non-NaN mask.
    Input infinities are outside this finite-or-NaN contract.

    Fit means/sample SD jointly on retained training and valid prediction rows,
    then normalize complete partitions. Constant columns retain source NaN/inf
    results; do not recompute validity after normalization. Return normalized
    train, labels, normalized prediction and validity. Labels map source -1 to
    model class0, with positives class1 first. No mutation or file discovery.
    """
    groups=[primary_positive,auxiliary_positive,negative,prediction]
    if any(not isinstance(group,list) for group in groups) or not prediction:
        raise ValueError('four ordered clip lists with nonempty prediction required')
    matrices=[]
    for group in groups:
        arrays=[np.asarray(x) for x in group]
        if any(x.dtype.kind not in 'iuf' or x.shape!=(19,1965) or np.any(np.isinf(x)) for x in arrays):
            raise ValueError('clip feature matrices must be real 19-by-1965 without infinities; NaN allowed')
        matrices.append(np.concatenate(arrays,axis=0).astype(float) if arrays else np.empty((0,1965)))
    retained=[x[~np.isnan(x).any(axis=1)] for x in matrices[:3]]
    positive_count=len(retained[0])+len(retained[1])
    negative_count=len(retained[2])
    if not positive_count or not negative_count:
        raise ValueError('both training classes must retain non-NaN rows')
    train=np.concatenate(retained,axis=0)
    labels=np.r_[np.ones(positive_count,dtype=np.int64),np.zeros(negative_count,dtype=np.int64)]
    pred=matrices[3]
    valid=~np.isnan(pred).any(axis=1)
    normalized_train,normalized_pred=andriy_joint_normalization(train,pred,valid)
    return normalized_train,labels,normalized_pred,valid
