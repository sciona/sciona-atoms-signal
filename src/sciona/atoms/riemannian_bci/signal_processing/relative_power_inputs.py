"""Pinned source configuration for the relative-log-power model branch."""
from sciona.ghost.registry import register_atom

CONFIG_PORTS = [('window_size', 'int'), ('hop_size', 'int'), ('frequency_bands', 'list'),
                ('fs', 'float'), ('fft_window', 'int'), ('overlap', 'float'),
                ('n_bags', 'int'), ('n_estimators', 'int')]


def witness_relative_power_configuration() -> tuple:
    return relative_power_configuration()


@register_atom(witness_relative_power_configuration)
def relative_power_configuration() -> tuple:
    """Return settings from the pinned relative-power feature/model YAML.

    Frame 20-second segments at 400 Hz without overlap; estimate Welch power
    with 512-sample FFT frames and .25 overlap, then fit ten bags of 500 trees.
    Band definitions are fresh lists on every call.
    """
    return (8000, 8000, [[.1, 4], [4, 8], [8, 15], [15, 30], [30, 90], [90, 170]],
            400., 512, .25, 10, 500)
