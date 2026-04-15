from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np
import scipy.integrate as scipy_integrate
from numpy.typing import ArrayLike

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_calculatebeatagreementsqi, witness_calculatecompositesqi_zz2018, witness_calculatefrequencypowersqi, witness_calculatekurtosissqi
from biosppy.signals.ecg import ZZ2018
from biosppy.signals.ecg import bSQI
from biosppy.signals.ecg import fSQI
from biosppy.signals.ecg import kSQI

# Witness functions should be imported from the generated witnesses module


def _ensure_scipy_trapz() -> None:
    """Compat shim for BioSPPy on SciPy versions without integrate.trapz."""
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    if not hasattr(scipy_integrate, "trapz"):
        scipy_integrate.trapz = np.trapz  # type: ignore[attr-defined]

@register_atom(witness_calculatecompositesqi_zz2018)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateCompositeSQI_ZZ2018 output must not be None")
def calculatecompositesqi_zz2018(
    signal: ArrayLike,
    detector_1: ArrayLike,
    detector_2: ArrayLike,
    fs: float,
    search_window: int,
    nseg: int,
    mode: str,
) -> float:
    """Calculates a composite Signal Quality Index (SQI) for a signal, using multiple detectors and parameters. This likely serves as an orchestrator or a specific implementation from a paper.

    Args:
        signal: Primary physiological signal waveform.
        detector_1: Array of beat detections from the first detector.
        detector_2: Array of beat detections from the second detector.
        fs: Sampling frequency of the signal.
        search_window: Window size for searching or comparison.
        nseg: Number of segments for spectral analysis.
        mode: Operational mode for the calculation.

    Returns:
        The final composite SQI score.
    """
    _ensure_scipy_trapz()
    return ZZ2018(signal=signal, detector_1=detector_1, detector_2=detector_2, fs=fs, search_window=search_window, nseg=nseg, mode=mode)

@register_atom(witness_calculatebeatagreementsqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateBeatAgreementSQI output must not be None")
def calculatebeatagreementsqi(
    detector_1: ArrayLike,
    detector_2: ArrayLike,
    fs: float,
    mode: str,
    search_window: int,
) -> float:
    """Calculates a beat-based Signal Quality Index (bSQI) based on the agreement between two beat detectors.

Args:
    detector_1: Array of beat detections from the first detector.
    detector_2: Array of beat detections from the second detector.
    fs: Sampling frequency of the signal.
    mode: Operational mode for the calculation.
    search_window: Window size for comparing detector outputs.

Returns:
    The beat agreement Signal Quality Index (SQI) score."""
    return bSQI(detector_1=detector_1, detector_2=detector_2, fs=fs, mode=mode, search_window=search_window)

@register_atom(witness_calculatefrequencypowersqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateFrequencyPowerSQI output must not be None")
def calculatefrequencypowersqi(
    ecg_signal: ArrayLike,
    fs: float,
    nseg: int,
    num_spectrum: ArrayLike,
    dem_spectrum: ArrayLike,
    mode: str,
) -> float:
    """Calculates a frequency-based Signal Quality Index (fSQI) using the power spectrum of the electrocardiogram (ECG) signal.

Args:
    ecg_signal: The ECG signal waveform.
    fs: Sampling frequency of the signal.
    nseg: Number of segments for spectral analysis.
    num_spectrum: Numerator of the spectral ratio.
    dem_spectrum: Denominator of the spectral ratio.
    mode: Operational mode for the calculation.

Returns:
    The frequency power Signal Quality Index (SQI) score."""
    _ensure_scipy_trapz()
    return fSQI(ecg_signal=ecg_signal, fs=fs, nseg=nseg, num_spectrum=num_spectrum, dem_spectrum=dem_spectrum, mode=mode)

@register_atom(witness_calculatekurtosissqi)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda fisher: fisher is not None, "fisher cannot be None")
@icontract.ensure(lambda result: result is not None, "CalculateKurtosisSQI output must not be None")
def calculatekurtosissqi(signal: ArrayLike, fisher: bool) -> float:
    """Calculates a Signal Quality Index (kSQI) based on the statistical kurtosis of the signal.

Args:
    signal: The input signal waveform.
    fisher: Flag to indicate if Fisher_primes definition of kurtosis is used.

Returns:
    The kurtosis-based Signal Quality Index (SQI) score."""
    return kSQI(signal=signal, fisher=fisher)
