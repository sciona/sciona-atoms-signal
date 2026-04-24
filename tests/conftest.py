from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"


def _install_sciona_test_stubs() -> None:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    import sciona  # type: ignore

    src_namespace = str(SRC_ROOT / "sciona")
    if src_namespace not in sciona.__path__:
        sciona.__path__.append(src_namespace)

    ghost_module = sys.modules.get("sciona.ghost")
    if ghost_module is None:
        ghost_module = types.ModuleType("sciona.ghost")
        ghost_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["sciona.ghost"] = ghost_module

    if "sciona.ghost.registry" not in sys.modules:
        registry_module = types.ModuleType("sciona.ghost.registry")
        registry: dict[str, object] = {}
        witnesses: dict[str, object] = {}

        def register_atom(witness):
            def decorator(func):
                registry[func.__name__] = func
                witnesses[func.__name__] = witness
                return func

            return decorator

        def list_registered():
            return sorted(registry)

        def get_witness(name: str):
            return witnesses[name]

        registry_module.REGISTRY = registry  # type: ignore[attr-defined]
        registry_module.register_atom = register_atom  # type: ignore[attr-defined]
        registry_module.list_registered = list_registered  # type: ignore[attr-defined]
        registry_module.get_witness = get_witness  # type: ignore[attr-defined]
        sys.modules["sciona.ghost.registry"] = registry_module

    if "sciona.ghost.abstract" not in sys.modules:
        abstract_module = types.ModuleType("sciona.ghost.abstract")

        class _AbstractBase:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class AbstractArray(_AbstractBase):
            pass

        class AbstractScalar(_AbstractBase):
            pass

        class AbstractSignal(_AbstractBase):
            pass

        class AbstractDistribution(_AbstractBase):
            pass

        abstract_module.AbstractArray = AbstractArray  # type: ignore[attr-defined]
        abstract_module.AbstractScalar = AbstractScalar  # type: ignore[attr-defined]
        abstract_module.AbstractSignal = AbstractSignal  # type: ignore[attr-defined]
        abstract_module.AbstractDistribution = AbstractDistribution  # type: ignore[attr-defined]
        sys.modules["sciona.ghost.abstract"] = abstract_module


_install_sciona_test_stubs()
