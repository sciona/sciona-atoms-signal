"""Runtime atoms for Graph Signal Processing expansion rules.

Provides deterministic, pure functions for graph spectral analysis
quality diagnostics:

  - Graph connectivity validation (disconnected components)
  - Laplacian symmetry check (numeric symmetry of L)
  - Spectral gap analysis (algebraic connectivity)
  - Graph filter frequency response validation (filter shape quality)
"""

from __future__ import annotations

import numpy as np
import icontract
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def witness_validate_graph_connectivity(
    adjacency: AbstractArray,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe connected-component count and connectedness flag."""
    return (
        AbstractScalar(dtype="int64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


def witness_check_laplacian_symmetry(
    laplacian: AbstractArray,
    tolerance: AbstractScalar,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe symmetry error and symmetry flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


def witness_analyze_spectral_gap(
    eigenvalues: AbstractArray,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe spectral gap and graph-connectivity quality flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


def witness_validate_filter_response(
    filter_response: AbstractArray,
    eigenvalues: AbstractArray,
) -> tuple[AbstractScalar, AbstractScalar]:
    """Describe filter gain and stability flag."""
    return (
        AbstractScalar(dtype="float64", min_val=0.0),
        AbstractScalar(dtype="bool"),
    )


# ---------------------------------------------------------------------------
# Graph connectivity validation
# ---------------------------------------------------------------------------


@register_atom(witness_validate_graph_connectivity)
@icontract.require(lambda adjacency: np.asarray(adjacency).ndim == 2, "adjacency must be 2-D")
@icontract.require(lambda adjacency: np.asarray(adjacency).shape[0] == np.asarray(adjacency).shape[1], "adjacency must be square")
@icontract.ensure(lambda result: result[0] >= 0, "component count must be non-negative")
def validate_graph_connectivity(
    adjacency: np.ndarray,
) -> tuple[int, bool]:
    """Validate that the graph is connected.

    Disconnected graphs produce zero eigenvalues in the Laplacian,
    which can cause division-by-zero in graph filters.

    Args:
        adjacency: (n, n) adjacency or weight matrix.

    Returns:
        (n_components, is_connected) where n_components is the number
        of connected components (via BFS) and is_connected is True
        if there is exactly 1 component.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return 0, False

    n = A.shape[0]
    if n == 0:
        return 0, True

    visited = np.zeros(n, dtype=bool)
    components = 0

    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            neighbors = np.nonzero(A[node] != 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    stack.append(nb)

    return components, components == 1


# ---------------------------------------------------------------------------
# Laplacian symmetry check
# ---------------------------------------------------------------------------


@register_atom(witness_check_laplacian_symmetry)
@icontract.require(lambda laplacian: np.asarray(laplacian).ndim == 2, "laplacian must be 2-D")
@icontract.require(lambda laplacian: np.asarray(laplacian).shape[0] == np.asarray(laplacian).shape[1], "laplacian must be square")
@icontract.require(lambda tolerance: tolerance >= 0.0, "tolerance must be non-negative")
@icontract.ensure(lambda result: result[0] >= 0.0, "asymmetry must be non-negative")
def check_laplacian_symmetry(
    laplacian: np.ndarray,
    tolerance: float = 1e-10,
) -> tuple[float, bool]:
    """Check that the graph Laplacian is symmetric.

    An asymmetric Laplacian indicates errors in graph construction
    or directed edges that require special handling.

    Args:
        laplacian: (n, n) Laplacian matrix.
        tolerance: max acceptable asymmetry.

    Returns:
        (max_asymmetry, is_symmetric) where max_asymmetry is
        max|L - L^T| and is_symmetric is True if below tolerance.
    """
    L = np.asarray(laplacian, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        return float("inf"), False

    if L.shape[0] == 0:
        return 0.0, True

    diff = np.abs(L - L.T)
    max_asym = float(np.max(diff))
    return max_asym, max_asym < tolerance


# ---------------------------------------------------------------------------
# Spectral gap analysis
# ---------------------------------------------------------------------------


@register_atom(witness_analyze_spectral_gap)
@icontract.require(lambda eigenvalues: np.asarray(eigenvalues).size >= 2, "at least two eigenvalues are required")
@icontract.require(lambda eigenvalues: bool(np.all(np.isfinite(np.asarray(eigenvalues, dtype=np.float64)))), "eigenvalues must be finite")
@icontract.ensure(lambda result: result[0] >= 0.0, "spectral gap must be non-negative")
def analyze_spectral_gap(
    eigenvalues: np.ndarray,
) -> tuple[float, bool]:
    """Analyze the spectral gap (algebraic connectivity) of the graph.

    The spectral gap is the second-smallest eigenvalue of the
    Laplacian.  A small spectral gap indicates near-disconnection
    or poor mixing properties.

    Args:
        eigenvalues: sorted eigenvalues of the graph Laplacian.

    Returns:
        (spectral_gap, is_well_connected) where spectral_gap is
        lambda_2 and is_well_connected is True if lambda_2 > 0.01.
    """
    eigs = np.sort(np.asarray(eigenvalues, dtype=np.float64).ravel())
    if len(eigs) < 2:
        return 0.0, False

    # lambda_2 is the second eigenvalue (first non-zero for connected graph)
    gap = float(eigs[1])
    return gap, gap > 0.01


# ---------------------------------------------------------------------------
# Graph filter frequency response validation
# ---------------------------------------------------------------------------


@register_atom(witness_validate_filter_response)
@icontract.require(lambda filter_response: np.asarray(filter_response).size > 0, "filter_response must be non-empty")
@icontract.require(lambda filter_response, eigenvalues: np.asarray(filter_response).size == np.asarray(eigenvalues).size, "response and eigenvalue counts must match")
@icontract.ensure(lambda result: result[0] >= 0.0, "max gain must be non-negative")
def validate_filter_response(
    filter_response: np.ndarray,
    eigenvalues: np.ndarray,
) -> tuple[float, bool]:
    """Validate that the graph filter response is well-behaved.

    Checks that the filter response is finite and bounded, which
    is necessary for stable graph filtering.

    Args:
        filter_response: 1-D array of filter values h(lambda_i).
        eigenvalues: corresponding graph Laplacian eigenvalues.

    Returns:
        (max_gain, is_stable) where max_gain is max|h(lambda)|
        and is_stable is True if all values are finite and max_gain < 100.
    """
    h = np.asarray(filter_response, dtype=np.float64).ravel()
    if len(h) == 0:
        return 0.0, True

    if not np.all(np.isfinite(h)):
        return float("inf"), False

    max_gain = float(np.max(np.abs(h)))
    return max_gain, max_gain < 100.0
