from __future__ import annotations
from sciona.ghost.abstract import AbstractArray


def witness_wrapperpredictionsignalcomputation(prediction: AbstractArray, raw_signal: AbstractArray) -> AbstractArray:
    """Describe the integer peak-index vector returned by Kazemi wrapper post-processing."""
    result = AbstractArray(
        shape=(0,),
        dtype="int64",
    )
    return result

def witness_signalarraynormalization(arr: AbstractArray) -> AbstractArray:
    """Describe the normalized array returned by the Kazemi normalization helper."""
    result = AbstractArray(
        shape=arr.shape,
        dtype="float64",
    )
    return result
