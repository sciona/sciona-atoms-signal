"""Explicit boundaries and pinned configuration for the two Riemannian branches."""
from sciona.ghost.registry import register_atom
from .segment_windows import window_signal_segments, witness_window_signal_segments


CONFIG_PORTS = [
    ('window_size', 'int'), ('hop_size', 'int'), ('delays', 'list'), ('subsample', 'int'),
    ('frequency_bands', 'list'), ('fs', 'float'), ('fft_window', 'int'), ('overlap', 'float'),
    ('autocorrelation_metric', 'str'), ('autocorrelation_update', 'bool'),
    ('coherence_metric', 'str'), ('coherence_update', 'bool'),
    ('autocorrelation_bags', 'int'), ('coherence_bags', 'int'), ('n_estimators', 'int'),
]


def witness_source_riemann_configuration() -> tuple:
    return source_riemann_configuration()


@register_atom(witness_source_riemann_configuration)
def source_riemann_configuration() -> tuple:
    """Return immutable-source settings as distinct graph ports, with fresh lists.

    Taken from the pinned public autocorrelation/coherence feature and model YAML.
    Windows are 20 seconds at 400 Hz, non-overlapping. This keeps different
    branch metrics, reference updates and bag counts off shared runtime inputs.
    """
    return (8000, 8000, [1, 2, 4, 8, 16, 32, 64], 2,
            [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]],
            400., 512, .5, 'identity', False, 'logeuclid', True, 4, 10, 500)


def witness_prepare_segment_windows(training_segments: list, prediction_segments: list,
                                     window_size: int, hop_size: int) -> tuple:
    train, train_ids = witness_window_signal_segments(training_segments, window_size, hop_size)
    prediction, prediction_ids = witness_window_signal_segments(prediction_segments, window_size, hop_size)
    return train, train_ids, prediction, prediction_ids, len(prediction_segments)


@register_atom(witness_prepare_segment_windows)
def prepare_segment_windows(training_segments: list, prediction_segments: list,
                             window_size: int, hop_size: int) -> tuple:
    """Window training/prediction separately and expose aligned segment identities.

    Return training windows/indices, prediction windows/indices and prediction
    segment count. Both collections must share channel count. Delegates framing
    to the registered multichannel primitive, with no training/prediction mixing.
    """
    train, train_ids = window_signal_segments(training_segments, window_size, hop_size)
    prediction, prediction_ids = window_signal_segments(prediction_segments, window_size, hop_size)
    if train.shape[1] != prediction.shape[1]:
        raise ValueError('training and prediction channel counts must match')
    return train, train_ids, prediction, prediction_ids, len(prediction_segments)
