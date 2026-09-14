# SPDX-License-Identifier: GPL-2.0-or-later
"""Full Andriy clip feature extraction from preprocessed signals."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .andriy_mainstream_features import andriy_documented_mainstream_features
from .andriy_ar_features import andriy_documented_ar_features
from .andriy_connectivity_features import andriy_connectivity_features
from .andriy_feature_assembly import andriy_assemble_clip_features


def witness_andriy_documented_clip_features(segment_256: AbstractArray, csp_segment_128: AbstractArray):
    count=segment_256.shape[1]//7680
    return (AbstractArray(shape=(count,1965),dtype='float64'),
            AbstractArray(shape=(count,),dtype='bool'))


@register_atom(witness_andriy_documented_clip_features)
def andriy_documented_clip_features(segment_256: NDArray[np.float64], csp_segment_128: NDArray[np.float64]) -> tuple[NDArray[np.float64],NDArray[np.bool_]]:
    """Extract windows/1965 features and validity from one preprocessed clip.

    Inputs are finite 16-channel 256-Hz samples and first-CSP-component 128-Hz
    samples (shape 1/samples). Require at least one complete 30-second window
    and equal complete-window counts. Drop trailing incomplete windows in each.
    Mainstream and plain AR skip channel/windows with range >=1000; CSP AR does
    not apply this gate. Connectivity masks only fully zero windows. Source
    low-range masks, within-clip imputation, selected logs, column ordering and
    non-NaN validity are preserved by the component providers.

    This composes declared current-runtime contracts: documented Welch and
    estimated-state AR settings, current SciPy connectivity filtering, and no
    masked-window RNG consumption. Historical MATLAB parity is unproven.
    Filtering/resampling, CSP fitting/projection, training selection and joint
    normalization happen outside this atom. Inputs are not mutated.
    """
    inputs=[np.asarray(segment_256),np.asarray(csp_segment_128)]
    for value,channels,size in zip(inputs,[16,1],[7680,3840]):
        if value.dtype.kind not in 'iuf' or value.ndim!=2 or value.shape[0]!=channels or value.shape[1]<size or not np.all(np.isfinite(value)):
            raise ValueError('finite 16-channel 256-Hz and one-channel 128-Hz clips with complete 30-second windows required')
    count=inputs[0].shape[1]//7680
    if inputs[1].shape[1]//3840!=count:
        raise ValueError('preprocessed and CSP clips must have equal complete-window counts')
    channels=inputs[0][:,:count*7680].astype(float).reshape(16,count,7680)
    flattened=channels.reshape(16*count,7680)
    eligible=np.ptp(flattened,axis=1)<1000
    main=np.full((len(flattened),102),np.nan)
    ar=np.full((len(flattened),9),np.nan)
    if np.any(eligible):
        main[eligible]=andriy_documented_mainstream_features(flattened[eligible])
        ar[eligible]=andriy_documented_ar_features(flattened[eligible])
    csp_windows=inputs[1][:,:count*3840].astype(float).reshape(count,3840)
    csp=andriy_documented_ar_features(csp_windows)[None,:,:]
    connectivity=andriy_connectivity_features(channels.transpose(1,0,2))
    return andriy_assemble_clip_features(main.reshape(16,count,102),ar.reshape(16,count,9),csp,connectivity)
