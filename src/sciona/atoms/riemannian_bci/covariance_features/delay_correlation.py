"""Per-channel circular-delay correlations for source-compatible BCI features."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _delay_list(delays):
    if isinstance(delays,(int,np.integer)) and not isinstance(delays,bool):
        if delays<2:raise ValueError('integer order must be at least two')
        return list(range(1,int(delays)))
    if not isinstance(delays,(list,tuple,np.ndarray)) or len(delays)==0:
        raise ValueError('nonempty positive delay list required')
    if any(isinstance(d,bool) or not isinstance(d,(int,np.integer)) or d<1 for d in delays):
        raise ValueError('positive integer delays required')
    if len(set(delays))!=len(delays):raise ValueError('distinct delays required')
    return [int(d) for d in delays]


def witness_channel_delay_correlations(windows: AbstractArray,delays: list|int,subsample: int=4) -> AbstractArray:
    if len(windows.shape)!=3:raise ValueError('window/channel/sample tensor required')
    size=len(_delay_list(delays))+1
    return AbstractArray(shape=(windows.shape[0],size,size,windows.shape[1]),dtype='float64')


@register_atom(witness_channel_delay_correlations)
def channel_delay_correlations(windows: NDArray[np.float64],delays: list|int,subsample: int=4) -> NDArray[np.float64]:
    """Return per-channel correlations of circularly delayed signal copies.

    Input shape is (windows, channels, samples). Subsample each channel first,
    then stack the original signal and np.roll(signal, delay) for each delay.
    Integer order k means delays 1..k-1; an explicit list retains its given order.
    Output shape is (windows, 1+len(delays), 1+len(delays), channels), matching the
    source AutoCorrMat layout. Values must be finite and every channel nonconstant.
    Delays must be distinct positive integers below the subsampled sample count.

    Correlations are positive semidefinite; strict positive definiteness is not
    guaranteed. No regularization is silently added. This is not the existing
    truncated-lag multichannel covariance feature. Source: AutoCorrMat in the
    pinned competition code and pyRiemann HankelCovariances (2016 implementation).
    """
    a=np.asarray(windows);lags=_delay_list(delays)
    if isinstance(subsample,bool) or not isinstance(subsample,(int,np.integer)) or subsample<1:
        raise ValueError('positive integer subsample required')
    if a.dtype.kind not in 'iuf' or a.ndim!=3 or min(a.shape)==0 or not np.all(np.isfinite(a)):
        raise ValueError('finite nonempty window/channel/sample tensor required')
    size=len(lags)+1
    result=np.empty((a.shape[0],size,size,a.shape[1]),dtype=float)
    for wi,window in enumerate(a):
        for ci,channel in enumerate(window):
            values=channel[::int(subsample)].astype(float)
            if max(lags)>=len(values) or np.std(values)==0:
                raise ValueError('nonconstant subsampled channel longer than all delays required')
            delayed=np.stack([values]+[np.roll(values,d) for d in lags])
            with np.errstate(all='raise'):
                correlation=np.corrcoef(delayed)
            if not np.all(np.isfinite(correlation)):raise ValueError('undefined delay correlation')
            result[wi,:,:,ci]=correlation
    return result
