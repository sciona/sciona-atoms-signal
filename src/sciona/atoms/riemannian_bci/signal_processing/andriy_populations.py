"""Keep the three source populations independent through feature preparation."""
from sciona.ghost.registry import register_atom
from .andriy_population import (
    andriy_documented_population, witness_andriy_documented_population,
)


def _ordered_inputs(*groups):
    if any(not isinstance(group, list) or len(group) != 3 for group in groups):
        raise ValueError('each input must contain exactly three ordered populations')
    return zip(*groups)


def witness_andriy_documented_populations(csp_candidates: list, candidate_sequences: list,
        primary_positive: list, auxiliary_positive: list, negative: list,
        prediction: list, sampling_frequencies: list) -> tuple:
    results = [witness_andriy_documented_population(*values) for values in _ordered_inputs(
        csp_candidates, candidate_sequences, primary_positive, auxiliary_positive,
        negative, prediction, sampling_frequencies)]
    return tuple(list(part) for part in zip(*results))


@register_atom(witness_andriy_documented_populations)
def andriy_documented_populations(csp_candidates: list, candidate_sequences: list,
        primary_positive: list, auxiliary_positive: list, negative: list,
        prediction: list, sampling_frequencies: list) -> tuple[list, list, list, list]:
    """Prepare three ordered populations independently for the source R branches.

    Every argument is a list of exactly three population inputs. Each entry
    follows andriy_documented_population's raw clip, membership and enumeration
    contract. Sampling frequency is common within each population. CSP fitting
    and joint normalization never mix populations. Preserve caller population
    order in four output lists: training features, positive-first binary labels,
    prediction features and validity masks. The model branches rank only after
    fitting and scoring each population separately. All underlying documented
    current-runtime limitations apply; no historical MATLAB equivalence claim.
    """
    results = [andriy_documented_population(*values) for values in _ordered_inputs(
        csp_candidates, candidate_sequences, primary_positive, auxiliary_positive,
        negative, prediction, sampling_frequencies)]
    return tuple(list(part) for part in zip(*results))
