from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_templatefeaturecomputation(hc: AbstractArray) -> AbstractArray:
    """Describe template-matching feature output for a beat collection."""
    return AbstractArray(shape=hc.shape, dtype="float64")
