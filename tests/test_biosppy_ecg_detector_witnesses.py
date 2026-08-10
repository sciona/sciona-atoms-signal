from __future__ import annotations

from sciona.atoms.signal_processing.biosppy.ecg_detectors.witnesses import (
    witness_hamilton_segmentation,
    witness_hamilton_segmenter,
)
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def test_hamilton_witnesses_return_bounded_peak_indices() -> None:
    signal = AbstractSignal(
        shape=(21600,),
        dtype="float64",
        sampling_rate=360.0,
        domain="time",
    )
    sampling_rate = AbstractScalar(dtype="float64", min_val=360.0, max_val=360.0)

    for witness in (witness_hamilton_segmentation, witness_hamilton_segmenter):
        peaks = witness(signal, sampling_rate)
        assert isinstance(peaks, AbstractArray)
        assert not isinstance(peaks, AbstractSignal)
        assert peaks.dtype == "int64"
        assert peaks.is_sorted
        assert peaks.min_val == 0
        assert peaks.max_val == 21599
