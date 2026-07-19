"""Experiment tracking (BUILD_PLAN T29).

**TensorBoard, not W&B.** It ships with torch, works offline and inside Docker,
needs no account, and makes no network call -- which matters on a college GPU
box that may not have outbound internet, and matters more for a project whose
data is under a research-only EULA. A solo project makes two runs; hosted sweeps
solve a problem we do not have.

Wrapped rather than used directly so that (a) a missing tensorboard install
degrades to a no-op instead of killing a training run, and (b) tests need no
writer at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tracker:
    """Thin SummaryWriter wrapper. No-ops when disabled or unavailable."""

    def __init__(self, run_dir: Path | str | None = None, enabled: bool = True) -> None:
        self.writer = None
        if not enabled or run_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            # Losing a training run because a *plotting* dependency is missing
            # would be absurd. Warn and carry on.
            logger.warning("tensorboard not installed -- metrics will only go to the log")
            return
        self.writer = SummaryWriter(log_dir=str(Path(run_dir) / "tb"))

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def metrics(self, prefix: str, metrics: Any, step: int) -> None:
        """Log a :class:`~ml.utils.metrics.Metrics` object.

        Flattens the per-manipulation breakdown into ``val/auc/NeuralTextures``
        style tags so the methods overlay on one TensorBoard chart -- which is
        the whole point of tracking them separately (T19/T28).
        """
        if self.writer is None:
            return
        import math

        for key in ("loss", "auc", "ap", "eer", "accuracy", "accuracy_at_eer"):
            value = getattr(metrics, key, float("nan"))
            if value is not None and not math.isnan(value):
                self.scalar(f"{prefix}/{key}", value, step)

        for method, entry in (getattr(metrics, "per_manipulation", None) or {}).items():
            for key, value in entry.items():
                if key != "n" and not math.isnan(value):
                    self.scalar(f"{prefix}/{key}/{method}", value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
