"""Source direct-FFT spectral feature block, excluding Welch-derived moments."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_andriy_spectral_features(windows: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(windows.shape[0], 82), dtype='float64')


@register_atom(witness_andriy_spectral_features)
def andriy_spectral_features(windows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return82 direct-FFT spectral features at fixed source256Hz.

    Order: total+31 overlapping2Hz powers (32); normalized2Hz powers starting
    at11..30 then0..10 (31); spectral edges90/95/80% (3); extra band powers
    [3,15],[15,30],[59,61],[51,69],[20,30],[59,61],[25,128] (7); their normalized
    values (7); peak frequency1..32Hz and spectral entropy (2).
    Repeated59..61Hz entries intentionally preserve source ordering.

    Spectrum is abs(fullFFT)^2/N restricted to nonnegative frequencies, without
    doubling. Power integrates bins strictly above lower endpoint through upper
    endpoint with unit-spacing trapezoids, not Hz spacing. Source peak/edge
    indexing returns one bin above the identified bin. Entropy uses eps inside
    normalization/log2. Inputs are finite windows/samples, length a positive
    multiple of256 so every integer endpoint lies exactly on the source grid.
    Zero total power yields NaN relative powers. No input mutation. Upstream
    low-range masking and Welch mean/bandwidth are separate operations.
    """
    x = np.asarray(windows)
    if x.dtype.kind not in 'iuf' or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 256 or x.shape[1] % 256 or not np.all(np.isfinite(x)):
        raise ValueError('finite windows/samples with length a positive multiple of256 required')
    n = x.shape[1]
    frequency = 256*np.arange(n//2+1)/n
    index = lambda hz: int(hz*n//256)
    extras = [(3,15),(15,30),(59,61),(51,69),(20,30),(59,61),(25,128)]
    rows = []
    for row in x.astype(float):
        transformed = np.fft.fft(row)
        spectrum = (transformed*np.conj(transformed)).real[:n//2+1]/n
        def power(lo, hi):
            return np.trapezoid(spectrum[index(lo)+1:index(hi)+1])
        total = power(0,128)
        bands = np.array([power(lo,lo+2) for lo in range(31)])
        extra = np.array([power(lo,hi) for lo,hi in extras])
        restricted = spectrum[index(1):index(32)+1]
        edges = []
        for percent in [.9,.95,.8]:
            count = 1
            threshold = percent*sum(restricted)
            while sum(restricted[:count]) < threshold and count <= len(restricted)-1:
                count += 1
            edges.append(frequency[index(1)+count])
        peak = frequency[index(1)+np.argmax(restricted)+1]
        eps = np.finfo(float).eps
        pdf = spectrum/(sum(spectrum)+eps)
        entropy = -sum(pdf*np.log2(pdf+eps))
        with np.errstate(divide='ignore', invalid='ignore'):
            rows.append(np.r_[total,bands,bands[np.r_[11:31,0:11]]/total,edges,extra,extra/total,peak,entropy])
    return np.array(rows)
