"""Channel-preserving assembly of the source combined-feature model inputs."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_concatenate_channel_features(relative_power: AbstractArray, ar_standard_errors: AbstractArray,
                                          basic_statistics: AbstractArray, fractal_features: AbstractArray) -> AbstractArray:
    blocks = [relative_power, ar_standard_errors, basic_statistics, fractal_features]
    return AbstractArray(shape=(*relative_power.shape[:2], sum(b.shape[2] for b in blocks)), dtype='float64')


@register_atom(witness_concatenate_channel_features)
def concatenate_channel_features(relative_power: NDArray[np.float64], ar_standard_errors: NDArray[np.float64],
                                  basic_statistics: NDArray[np.float64], fractal_features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Join feature families along the final axis, retaining window/channel order.

    All four inputs are finite nonempty (windows, channels, features) tensors
    with identical first two axes and already aligned window/channel identities.
    Family order is relative power, AR coefficient standard errors, basic stats,
    then PFD/HFD/Hurst, matching source model datasets order. The configured
    source uses 6+6+6+3 features per channel. Concatenate before vectorization:
    flattening the individual families first would change classifier column order.
    Output is an independent float64 tensor; identities are supplied by upstream
    common windowing and cannot be inferred from numerical arrays alone.
    """
    blocks = [np.asarray(b) for b in [relative_power, ar_standard_errors, basic_statistics, fractal_features]]
    for block in blocks:
        if block.dtype.kind not in 'iuf' or block.ndim != 3 or min(block.shape) == 0 or not np.all(np.isfinite(block)):
            raise ValueError('finite nonempty window/channel/feature tensors required')
        if block.shape[:2] != blocks[0].shape[:2]:
            raise ValueError('matching window and channel axes required')
    return np.concatenate(blocks, axis=-1).astype(float, copy=False)
