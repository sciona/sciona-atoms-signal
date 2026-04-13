from __future__ import annotations

from typing import TYPE_CHECKING, Any

import icontract
import numpy as np
from ageoa.ghost.registry import register_atom

from .._vendor import load_e2e_ppg_module
from .witnesses import witness_gan_reconstruction, witness_generatereconstructedppg

if TYPE_CHECKING:
    import torch


@register_atom(witness_generatereconstructedppg)
@icontract.require(
    lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)),
    "sampling_rate must be numeric",
)
@icontract.ensure(lambda result: result is not None, "GenerateReconstructedPPG output must not be None")
def generatereconstructedppg(
    ppg_clean: np.ndarray | "torch.Tensor",
    noise: np.ndarray | "torch.Tensor",
    sampling_rate: int | float,
    generator: "torch.nn.Module",
    device: str | "torch.device",
) -> np.ndarray | "torch.Tensor":
    module = load_e2e_ppg_module("ppg_reconstruction")
    return module.gan_rec(
        ppg_clean=ppg_clean,
        noise=noise,
        sampling_rate=sampling_rate,
        generator=generator,
        device=device,
    )


@register_atom(witness_gan_reconstruction)
@icontract.require(
    lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)),
    "sampling_rate must be numeric",
)
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "gan_reconstruction output must not be None")
def gan_reconstruction(
    ppg_clean: np.ndarray,
    noise: list[int] | np.ndarray,
    sampling_rate: int,
    generator: Any,
    device: Any,
) -> Any:
    module = load_e2e_ppg_module("ppg_reconstruction")
    return module.gan_rec(
        ppg_clean=ppg_clean,
        noise=noise,
        sampling_rate=sampling_rate,
        generator=generator,
        device=device,
    )
