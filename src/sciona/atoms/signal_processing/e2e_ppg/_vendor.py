from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def resolve_e2e_ppg_root() -> Path:
    env_root = str(os.environ.get("SCIONA_E2E_PPG_ROOT", "") or "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "third_party" / "E2E-PPG")
        candidates.append(parent / "ageo-atoms" / "third_party" / "E2E-PPG")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise ModuleNotFoundError(
        "Could not locate the vendored E2E-PPG checkout. "
        "Set SCIONA_E2E_PPG_ROOT or place it under third_party/E2E-PPG."
    )


def load_e2e_ppg_module(module_name: str):
    root = resolve_e2e_ppg_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(module_name)


def with_reconstruction_model_compat(callback):
    module = load_e2e_ppg_module("ppg_reconstruction")
    import torch

    original_model_path = getattr(module, "MODEL_PATH", None)
    original_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        module.MODEL_PATH = str(resolve_e2e_ppg_root() / "models")
        torch.load = _compat_torch_load
        return callback(module)
    finally:
        torch.load = original_torch_load
        if original_model_path is not None:
            module.MODEL_PATH = original_model_path
