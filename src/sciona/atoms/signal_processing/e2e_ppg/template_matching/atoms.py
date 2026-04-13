from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from ageoa.ghost.registry import register_atom

from .._vendor import load_e2e_ppg_module
from .witnesses import witness_templatefeaturecomputation


@register_atom(witness_templatefeaturecomputation)  # type: ignore[untyped-decorator]
@icontract.require(lambda hc: hc is not None, "hc cannot be None")
@icontract.ensure(lambda result: result is not None, "TemplateFeatureComputation output must not be None")
def templatefeaturecomputation(
    hc: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[float, float]:
    module = load_e2e_ppg_module("ppg_sqa")
    normalized_hc = [np.asarray(beat, dtype=float) for beat in hc]
    tm_ave_eu, tm_ave_corr = module.template_matching_features(hc=normalized_hc)
    return float(tm_ave_eu), float(tm_ave_corr)
