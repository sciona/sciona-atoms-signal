from __future__ import annotations

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .._vendor import load_e2e_ppg_module
from .witnesses import witness_detect_heart_cycles, witness_heart_cycle_detection


@register_atom(witness_detect_heart_cycles)
@icontract.require(lambda ppg: isinstance(ppg, np.ndarray), "ppg must be a numpy array")
@icontract.require(
    lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)),
    "sampling_rate must be numeric",
)
@icontract.ensure(lambda result: result is not None, "detect_heart_cycles output must not be None")
def detect_heart_cycles(ppg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Detect heart-cycle boundaries in a PPG waveform."""
    module = load_e2e_ppg_module("ppg_sqa")
    return np.asarray(module.heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate))


@register_atom(witness_heart_cycle_detection)
@icontract.require(lambda ppg: isinstance(ppg, np.ndarray), "ppg must be a numpy array")
@icontract.require(
    lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)),
    "sampling_rate must be numeric",
)
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "heart_cycle_detection output must not be None")
def heart_cycle_detection(ppg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Alias the upstream heart-cycle detector under a descriptive public name."""
    module = load_e2e_ppg_module("ppg_sqa")
    return np.asarray(module.heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate))
