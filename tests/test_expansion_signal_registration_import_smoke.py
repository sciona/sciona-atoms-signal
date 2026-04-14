from __future__ import annotations

import importlib

from sciona.ghost.registry import list_registered


_MODULES_AND_SYMBOLS = {
    "sciona.atoms.expansion.graph_signal_processing": (
        "validate_graph_connectivity",
        "check_laplacian_symmetry",
        "analyze_spectral_gap",
        "validate_filter_response",
    ),
    "sciona.atoms.expansion.signal_detect_measure": (
        "estimate_snr",
        "analyze_peak_threshold_sensitivity",
        "check_event_rate_stationarity",
        "estimate_false_positive_rate",
    ),
    "sciona.atoms.expansion.signal_filter": (
        "analyze_pole_stability",
        "measure_passband_ripple",
        "analyze_group_delay_variation",
        "detect_transient_response",
    ),
    "sciona.atoms.expansion.signal_transform": (
        "analyze_window_leakage",
        "detect_spectral_aliasing",
        "validate_parseval_energy",
        "check_inverse_reconstruction",
    ),
}


def test_signal_expansion_registration_import_smoke() -> None:
    for module_name, symbols in _MODULES_AND_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for symbol in symbols:
            assert hasattr(module, symbol)

    registered = set(list_registered())
    expected = {symbol for symbols in _MODULES_AND_SYMBOLS.values() for symbol in symbols}
    assert expected <= registered
