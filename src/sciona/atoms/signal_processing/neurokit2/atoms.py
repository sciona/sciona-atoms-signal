from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from typing import Any

import numpy as np
import icontract

from ageoa.ghost.registry import register_atom
from neurokit2.ecg.ecg_quality import _ecg_quality_averageQRS, _ecg_quality_zhao2018

from .witnesses import witness_averageqrstemplate, witness_zhao2018hrvanalysis


@register_atom(witness_zhao2018hrvanalysis)
@icontract.require(lambda ecg_cleaned: ecg_cleaned is not None, "ecg_cleaned cannot be None")
@icontract.require(lambda rpeaks: rpeaks is None or isinstance(rpeaks, np.ndarray), "rpeaks must be an ndarray when provided")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (int, float, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda window: isinstance(window, (int, float, tuple)), "window must be numeric or tuple")
@icontract.require(lambda mode: isinstance(mode, str), "mode must be a string")
@icontract.ensure(lambda result: result is not None, "Zhao2018HRVAnalysis output must not be None")
def zhao2018hrvanalysis(
    ecg_cleaned: np.ndarray,
    rpeaks: np.ndarray | None = None,
    sampling_rate: int | float = 1000,
    window: int | float | tuple[Any, ...] = 1024,
    mode: str = "simple",
) -> str:
    """Compute the Zhao 2018 ECG quality verdict over a cleaned ECG trace."""
    return _ecg_quality_zhao2018(
        ecg_cleaned=ecg_cleaned,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        window=window,
        mode=mode,
    )


@register_atom(witness_averageqrstemplate)
@icontract.require(lambda ecg_cleaned: ecg_cleaned is not None, "ecg_cleaned cannot be None")
@icontract.require(lambda rpeaks: rpeaks is None or isinstance(rpeaks, np.ndarray), "rpeaks must be an ndarray when provided")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (int, float, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "AverageQRSTemplate output must not be None")
def averageqrstemplate(
    ecg_cleaned: np.ndarray,
    rpeaks: np.ndarray | None = None,
    sampling_rate: int | float = 1000,
) -> np.ndarray:
    """Build an average QRS template aligned to the detected R-peaks."""
    return _ecg_quality_averageQRS(
        ecg_cleaned=ecg_cleaned,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
    )
