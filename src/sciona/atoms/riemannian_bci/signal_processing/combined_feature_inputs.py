"""Pinned configuration for the source combined-feature model."""
from sciona.ghost.registry import register_atom

CONFIG_PORTS = [('window_size', 'int'), ('hop_size', 'int'), ('frequency_bands', 'list'),
                ('fs', 'float'), ('fft_window', 'int'), ('overlap', 'float'),
                ('n_bags', 'int'), ('n_estimators', 'int'), ('order', 'int'), ('subsample', 'int')]


def witness_combined_feature_configuration() -> tuple:
    return combined_feature_configuration()


@register_atom(witness_combined_feature_configuration)
def combined_feature_configuration() -> tuple:
    """Return fixed source feature/model settings with fresh band lists.

    Four feature families share non-overlapping 20-second windows at 400 Hz.
    Relative power uses FFT512/.25 overlap; AR standard errors use order5 and
    subsample4. The combined model uses five bags of 500 boosted trees.
    """
    return (8000, 8000, [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]],
            400., 512, .25, 5, 500, 5, 4)
