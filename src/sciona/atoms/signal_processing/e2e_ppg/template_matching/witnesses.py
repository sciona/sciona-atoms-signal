from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray


def witness_templatefeaturecomputation(hc: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=hc.shape, dtype="float64")
