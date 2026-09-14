"""Explicit family boundary and eleven-output packing for ensemble composition."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.registry import register_atom
from .ensemble_partitions import _groups

def witness_ensemble_family_inputs(alex_training_populations: list, alex_prediction_populations: list, alex_label_populations: list, alex_identity_populations: list, feng_training_populations: list, feng_prediction_populations: list, feng_label_populations: list, feng_identity_populations: list, prediction: list, andriy_identity_populations: list) -> tuple:
    return (alex_training_populations, alex_prediction_populations, alex_label_populations, alex_identity_populations, feng_training_populations, feng_prediction_populations, feng_label_populations, feng_identity_populations, prediction, andriy_identity_populations)


@register_atom(witness_ensemble_family_inputs)
def ensemble_family_inputs(alex_training_populations: list, alex_prediction_populations: list, alex_label_populations: list, alex_identity_populations: list, feng_training_populations: list, feng_prediction_populations: list, feng_label_populations: list, feng_identity_populations: list, prediction: list, andriy_identity_populations: list) -> tuple:
    """Expose distinct family input ports without converting or mixing raw groups.

    Each input has three ordered populations. Alex/Feng labels and identities
    are independently supplied. `prediction` contains Andriy raw prediction
    clip groups, paired with its separately supplied identity groups. Numerical
    and membership contracts are checked by downstream family providers.
    """
    _groups(alex_training_populations, alex_prediction_populations, alex_label_populations, alex_identity_populations, feng_training_populations, feng_prediction_populations, feng_label_populations, feng_identity_populations, prediction, andriy_identity_populations)
    return (alex_training_populations, alex_prediction_populations, alex_label_populations, alex_identity_populations, feng_training_populations, feng_prediction_populations, feng_label_populations, feng_identity_populations, prediction, andriy_identity_populations)


def witness_pack_eleven_predictions(combined: NDArray[np.float64], autocorrelation: NDArray[np.float64], coherence: NDArray[np.float64], relative_power: NDArray[np.float64], feng_xgb: NDArray[np.float64], feng_knn: NDArray[np.float64], feng_expanded_knn: NDArray[np.float64], feng_glm: NDArray[np.float64], andriy_svm: NDArray[np.float64], andriy_glm: NDArray[np.float64], andriy_xgb: NDArray[np.float64], alex_ids: NDArray[np.int64], feng_ids: NDArray[np.int64], andriy_ids: NDArray[np.int64]) -> tuple:
    return ([combined, autocorrelation, coherence, relative_power, feng_xgb, feng_knn, feng_expanded_knn, feng_glm, andriy_svm, andriy_glm, andriy_xgb], [alex_ids]*4 + [feng_ids]*4 + [andriy_ids]*3)


@register_atom(witness_pack_eleven_predictions)
def pack_eleven_predictions(combined: NDArray[np.float64], autocorrelation: NDArray[np.float64], coherence: NDArray[np.float64], relative_power: NDArray[np.float64], feng_xgb: NDArray[np.float64], feng_knn: NDArray[np.float64], feng_expanded_knn: NDArray[np.float64], feng_glm: NDArray[np.float64], andriy_svm: NDArray[np.float64], andriy_glm: NDArray[np.float64], andriy_xgb: NDArray[np.float64], alex_ids: NDArray[np.int64], feng_ids: NDArray[np.int64], andriy_ids: NDArray[np.int64]) -> tuple[list, list]:
    """Pack model outputs in the source blend order with matching family IDs.

    Each family must preserve a common prediction identity/order across its
    branches. Population collectors establish that order before this node.
    No ranking or numerical transformation occurs here. The final identity-aware
    blend validates lengths, coverage, uniqueness and finite model scores.
    """
    return ([combined, autocorrelation, coherence, relative_power, feng_xgb, feng_knn, feng_expanded_knn, feng_glm, andriy_svm, andriy_glm, andriy_xgb], [alex_ids]*4 + [feng_ids]*4 + [andriy_ids]*3)
