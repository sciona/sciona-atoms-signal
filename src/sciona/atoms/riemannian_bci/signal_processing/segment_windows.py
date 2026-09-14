"""Window multichannel segments with explicit segment provenance."""
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from sciona.atoms.audio_speech.atoms import audio_windows


def witness_window_signal_segments(segments: list,window_size: int,hop_size: int) -> tuple:
    if not segments or window_size<1 or hop_size<1:raise ValueError('nonempty segments and positive frame geometry required')
    channels=segments[0].shape[0]
    count=sum((s.shape[1]-window_size)//hop_size+1 for s in segments)
    return AbstractArray(shape=(count,channels,window_size),dtype='float64'),AbstractArray(shape=(count,),dtype='int64')


@register_atom(witness_window_signal_segments)
def window_signal_segments(segments: list,window_size: int,hop_size: int) -> tuple:
    """Return (windows, segment_indices) from multichannel observation segments.

    Segments are finite real matrices (channels, samples) sharing channel count;
    sample counts may differ. Window and hop are positive integers in samples.
    Only complete windows are emitted, in segment order then time order; trailing
    samples are dropped. Setting hop_size==window_size reproduces non-overlapping
    source segmentation. Every segment must contain at least one complete window.
    Returns independent float64 windows (windows, channels, window_size) and an
    int64 segment-index vector. Does not infer sampling rate or mutate inputs.
    """
    if not segments:raise ValueError('nonempty segments required')
    for value in [window_size,hop_size]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:raise ValueError('positive integer window and hop required')
    windows=[];indices=[];channels=None
    for index,segment in enumerate(segments):
        a=np.asarray(segment)
        if a.dtype.kind not in 'iuf' or a.ndim!=2 or a.shape[0]==0 or a.shape[1]<window_size or not np.all(np.isfinite(a)):
            raise ValueError('finite multichannel segment covering a full window required')
        if channels is None:channels=a.shape[0]
        if a.shape[0]!=channels:raise ValueError('consistent channel count required')
        per_channel=[audio_windows(channel,int(window_size),int(hop_size)) for channel in a.astype(np.float64)]
        block=np.stack(per_channel,axis=1)
        windows.append(block);indices.extend([index]*len(block))
    return np.concatenate(windows,axis=0),np.asarray(indices,dtype=np.int64)
