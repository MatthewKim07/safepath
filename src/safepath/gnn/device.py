"""Pick the training device: CUDA (also AMD ROCm builds), then Apple MPS, then CPU."""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)

DEVICE_ENV = "SAFEPATH_DEVICE"


def pick_device(requested: str | None = None) -> torch.device:
    """`requested` (or $SAFEPATH_DEVICE) wins if given; otherwise cuda -> mps -> cpu.

    Ops MPS lacks fall back to CPU when PYTORCH_ENABLE_MPS_FALLBACK=1, which is set here
    before any MPS work so training never fails on a missing kernel.
    """
    name = requested or os.environ.get(DEVICE_ENV) or "auto"
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = torch.device(name)
    log.info("training device: %s", device)
    return device
