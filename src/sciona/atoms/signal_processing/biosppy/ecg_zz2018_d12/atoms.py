from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np
import scipy.integrate as scipy_integrate
from numpy.typing import ArrayLike

import icontract
from ageoa.ghost.registry import register_atom
from biosppy.signals.ecg import ZZ2018, bSQI, fSQI, kSQI

from .witnesses import (
    witness_assemblezz2018sqi,
    witness_computebeatagreementsqi,
    witness_computefrequencysqi,
    witness_computekurtosissqi,
)


def _ensure_scipy_trapz() -> None:
    """Compat shim for BioSPPy on SciPy versions without integrate.trapz."""
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    if not hasattr(scipy_integrate, "trapz"):
        scipy_integrate.trapz = np.trapz  # type: ignore[attr-defined]


@register_atom(witness_computebeatagreementsqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.require(lambda search_window: isinstance(search_window, (float, int, np.number)), "search_window must be numeric")
@icontract.ensure(lambda result: result is not None, "ComputeBeatAgreementSQI output must not be None")
def computebeatagreementsqi(
    detector_1: ArrayLike,
    detector_2: ArrayLike,
    fs: float = 1000.0,
    mode: str = "simple",
    search_window: int = 150,
) -> float:
    """Compute the detector-agreement SQI from two beat-location streams."""
    return bSQI(detector_1=detector_1, detector_2=detector_2, fs=fs, mode=mode, search_window=search_window)


@register_atom(witness_computefrequencysqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "ComputeFrequencySQI output must not be None")
def computefrequencysqi(
    ecg_signal: ArrayLike,
    fs: float = 1000.0,
    nseg: int = 1024,
    num_spectrum: tuple[float, float] | ArrayLike = (5.0, 20.0),
    dem_spectrum: tuple[float, float] | ArrayLike | None = None,
    mode: str = "simple",
) -> float:
    """Compute the frequency-domain SQI for an ECG waveform."""
    _ensure_scipy_trapz()
    return fSQI(
        ecg_signal=ecg_signal,
        fs=fs,
        nseg=nseg,
        num_spectrum=num_spectrum,
        dem_spectrum=dem_spectrum,
        mode=mode,
    )


@register_atom(witness_computekurtosissqi)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda fisher: fisher is not None, "fisher cannot be None")
@icontract.ensure(lambda result: result is not None, "ComputeKurtosisSQI output must not be None")
def computekurtosissqi(signal: ArrayLike, fisher: bool = True) -> float:
    """Compute the kurtosis-based SQI for an ECG waveform."""
    return kSQI(signal=signal, fisher=fisher)


@register_atom(witness_assemblezz2018sqi)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda detector_1: detector_1 is not None, "detector_1 cannot be None")
@icontract.require(lambda detector_2: detector_2 is not None, "detector_2 cannot be None")
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.require(lambda search_window: isinstance(search_window, (float, int, np.number)), "search_window must be numeric")
@icontract.require(lambda nseg: isinstance(nseg, (float, int, np.number)), "nseg must be numeric")
@icontract.ensure(lambda result: result is not None, "AssembleZZ2018SQI output must not be None")
def assemblezz2018sqi(
    signal: ArrayLike,
    detector_1: ArrayLike,
    detector_2: ArrayLike,
    fs: float = 1000.0,
    search_window: int = 100,
    nseg: int = 1024,
    mode: str = "simple",
) -> str:
    """Run the composite ZZ2018 ECG quality classifier.

    Args:
        signal: Input ECG waveform in millivolts.
        detector_1: R-peak indices from the first detector.
        detector_2: R-peak indices from the second detector.
        fs: Sampling frequency in Hz.
        search_window: Matching window around candidate peaks in milliseconds.
        nseg: Frequency-axis resolution used by the spectral SQIs.
        mode: BioSPPy classifier mode, typically ``simple`` or ``fuzzy``.

    Returns:
        noise_class: Composite ZZ2018 quality class string.
    """
    _ensure_scipy_trapz()
    return ZZ2018(
        signal=signal,
        detector_1=detector_1,
        detector_2=detector_2,
        fs=fs,
        search_window=search_window,
        nseg=nseg,
        mode=mode,
    )
