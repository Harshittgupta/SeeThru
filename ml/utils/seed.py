"""Seeding and determinism (BUILD_PLAN T26).

The headline constraint, which is not optional:

**Do NOT enable ``torch.use_deterministic_algorithms(True)`` by default.** cuDNN's
BiLSTM backward has no deterministic implementation, so the video model (whose
whole temporal branch is a BiLSTM) raises the moment it tries to backprop::

    RuntimeError: cudnn RNN backward can only be called in training mode

So determinism is a flag, defaulted off, enabled for smoke runs where the model
is small and reproducibility beats speed.

Second trap: DataLoader workers. Each worker gets its own RNG, and without an
explicit ``worker_init_fn`` + ``generator`` the augmentation stream is not
reproducible across runs even with a fixed global seed.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch, and optionally force deterministic kernels.

    Args:
        seed: The seed.
        deterministic: Force deterministic algorithms. **Costs speed, and RAISES
            on the video model** -- cuDNN has no deterministic BiLSTM backward.
            Use for smoke tests and debugging, not for real runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Required by cuBLAS for deterministic matmuls; must be set before the
        # first CUDA call, so setting it here is best-effort if CUDA is already up.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Determinism ON (seed=%d). Slower; BiLSTM may warn.", seed)
    else:
        # benchmark=True lets cuDNN autotune conv algorithms for the input shape.
        # Ours is fixed at 224x224, so this is a free speedup after the first
        # few iterations.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info("Seeded (seed=%d). cudnn.benchmark=True.", seed)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible seed.

    PyTorch seeds ``torch`` per worker but leaves ``random`` and ``numpy``
    derived from the base seed in a way that is easy to get wrong. albumentations
    uses both, so without this the augmentation stream is not reproducible.
    """
    base = torch.initial_seed() % 2**32
    seed = (base + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded generator for DataLoader's shuffle order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
