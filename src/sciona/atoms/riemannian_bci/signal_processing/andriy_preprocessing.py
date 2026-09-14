"""Fixed-duration preprocessing and source CSP candidate selection."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray, AbstractMatrix
from sciona.ghost.registry import register_atom
from .andriy_filter import andriy_filter_trim
from .andriy_resampling import andriy_documented_resample


def witness_andriy_documented_preprocess_clip(segment: AbstractArray,sampling_frequency: int):
    remaining=segment.shape[1]-8*sampling_frequency+1
    return (AbstractArray(shape=(16,(remaining*256+sampling_frequency-1)//sampling_frequency),dtype='float64'),
            AbstractArray(shape=(16,(remaining*128+sampling_frequency-1)//sampling_frequency),dtype='float64'))


@register_atom(witness_andriy_documented_preprocess_clip)
def andriy_documented_preprocess_clip(segment: NDArray[np.float64],sampling_frequency: int) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Preprocess a finite 16-channel, exactly 600-second source clip at both rates.

    Center/notch/highpass once, preserve inclusive four-second edge trimming,
    and independently resample to 256 and 128 Hz. Both outputs contain nineteen
    complete 30-second windows plus the original residual tail. Uses the
    explicitly documented FIR design, not proven historical MATLAB defaults.
    The scalar input rate must be an integer >120 Hz. No input mutation.
    """
    x=np.asarray(segment)
    if isinstance(sampling_frequency,(bool,np.bool_)) or not isinstance(sampling_frequency,(int,np.integer)) or sampling_frequency<=120:
        raise ValueError('integer source sampling frequency above 120 Hz required')
    if x.ndim!=2 or x.shape!=(16,600*sampling_frequency):
        raise ValueError('exactly 600 seconds of 16-channel samples required')
    filtered=andriy_filter_trim(x,sampling_frequency)
    return (andriy_documented_resample(filtered,sampling_frequency,256),
            andriy_documented_resample(filtered,sampling_frequency,128))


def witness_andriy_csp_training_classes(positive_candidates,sequences):
    # Candidate counts depend on sequence and numeric range; dimensions are
    # intentionally not inferred from uninspected candidate values.
    return (AbstractMatrix(shape=('16','sequence6_samples'),dtype='float64'),AbstractMatrix(shape=('16','sequence1_samples'),dtype='float64'))


@register_atom(witness_andriy_csp_training_classes)
def andriy_csp_training_classes(positive_candidates: list,sequences: NDArray[np.int64]) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Select/concatenate source CSP classes from ordered 128-Hz positive clips.

    Caller supplies the ordered list corresponding to the source positive-file
    enumeration, including its first two entries, and sequence positions 1..6.
    Preserve the source's unconditional first-two-entry skip, select positions
    1 and 6, and accept only clips with strictly positive range in every channel.
    Return sequence6 concatenation first (source preD), sequence1 second (intD).
    Both originate from the positive candidate pool; intD here does not mean
    independently supplied negative-labelled training examples. File discovery
    and source ordering are caller responsibilities, not inferred from names.

    Inputs are finite 16-channel matrices with at least one complete 30-second
    128-Hz window; tails are retained for fitting. Require both selected classes.
    Preserve clip/sample order and return independent arrays without mutation.
    """
    seq=np.asarray(sequences)
    if not isinstance(positive_candidates,list) or len(positive_candidates)<4 or seq.dtype.kind not in 'iu' or seq.shape!=(len(positive_candidates),) or np.any((seq<1)|(seq>6)):
        raise ValueError('ordered positive candidates and matching sequence positions 1..6 required')
    arrays=[np.asarray(value) for value in positive_candidates]
    if any(x.dtype.kind not in 'iuf' or x.ndim!=2 or x.shape[0]!=16 or x.shape[1]<3840 or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError('finite 16-channel preprocessed candidate clips with at least 3840 samples required')
    classes={1:[],6:[]}
    for x,position in zip(arrays[2:],seq[2:]):
        if position in classes and np.min(np.ptp(x,axis=1))>0:
            classes[int(position)].append(x.astype(float))
    if not all(classes.values()):
        raise ValueError('source CSP selection must retain both sequence1 and sequence6 classes')
    return np.concatenate(classes[6],axis=1),np.concatenate(classes[1],axis=1)
