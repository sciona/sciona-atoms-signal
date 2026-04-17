from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def witness_templatefeaturecomputation(hc: AbstractArray) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe template-matching feature output for a beat collection."""
    return (AbstractScalar(dtype="float64"), AbstractScalar(dtype="float64"))
