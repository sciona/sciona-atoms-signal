from __future__ import annotations

from importlib import import_module

from sciona.probes.signal_processing.biosppy_online_filter import probe_records as online_filter_probe_records
from sciona.probes.signal_processing.biosppy_online_filter_codex import probe_records as online_filter_codex_probe_records
from sciona.probes.signal_processing.biosppy_online_filter_v2 import probe_records as online_filter_v2_probe_records


def _assert_probe_records_resolve(probe_records: list[dict[str, object]]) -> None:
    for record in probe_records:
        module = import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))
        fqdn_parts = str(record["atom_fqdn"]).split(".")
        imported = import_module(".".join(fqdn_parts[:-1]))
        assert getattr(imported, fqdn_parts[-1]) is getattr(
            module,
            str(record["wrapper_symbol"]),
        )


def test_online_filter_probe_records_resolve_to_live_symbols() -> None:
    _assert_probe_records_resolve(online_filter_probe_records())


def test_online_filter_codex_probe_records_resolve_to_live_symbols() -> None:
    _assert_probe_records_resolve(online_filter_codex_probe_records())


def test_online_filter_v2_probe_records_resolve_to_live_symbols() -> None:
    _assert_probe_records_resolve(online_filter_v2_probe_records())


def test_online_filter_probe_records_publish_expected_symbols() -> None:
    assert {str(record["wrapper_symbol"]) for record in online_filter_probe_records()} == {
        "filterstateinit",
        "filterstep",
    }
    assert {str(record["wrapper_symbol"]) for record in online_filter_codex_probe_records()} == {
        "filterstateinit",
        "filterstep",
    }
    assert {str(record["wrapper_symbol"]) for record in online_filter_v2_probe_records()} == {
        "filterstateinit",
        "filterstep",
    }
