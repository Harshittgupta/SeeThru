"""Training/eval loop internals for SEETHRU (BUILD_PLAN T30).

No CLI, no file IO, no checkpointing -- ``ml/train.py`` owns those. This module
owns one epoch of training and one pass of evaluation, so both can be tested
without a filesystem.

The four things here that are easy to get wrong, and expensive:

1. **bf16 over fp16 where available.** ``frequency_branch``'s
   ``log(|fft| + 1e-8)`` and ``AttentionFusion``'s ``-inf`` mask are exponent-range
   hazards; fp16's range is ~6e-5 to 65504 and a log of a small magnitude
   underflows. bf16 has fp32's exponent range and needs no loss scaling at all.
2. **The FFT must run in fp32.** autocast already lists ``fft2`` as fp32-only, but
   relying on an implicit allowlist for a numerical hazard is how it breaks
   silently on a torch upgrade. We say so explicitly.
3. **Freezing must call ``.eval()``, not just ``requires_grad_(False)``.**
   ``requires_grad=False`` does NOT stop EfficientNet's BatchNorms updating their
   running statistics, so a "frozen" backbone silently drifts for two epochs and
   the ImageNet features you meant to preserve are quietly corrupted.
4. **Unscale before clipping.** Under fp16 the gradients are scaled; clipping them
   pre-unscale clips the wrong magnitude, which either does nothing or destroys
   the step depending on the scale factor.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.utils.metrics import Metrics, compute_metrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Device / precision
# --------------------------------------------------------------------------- #
def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def resolve_amp(
    enabled: bool, dtype_spec: str, device: torch.device
) -> tuple[bool, torch.dtype | None]:
    """Decide the autocast dtype. → ``(enabled, dtype)``.

    bf16 when the GPU supports it (Ampere+), else fp16. On CPU, autocast buys
    nothing for this model and complicates debugging, so it is off -- which also
    means the smoke test (T34) runs in plain fp32 and any NaN is attributable to
    the loop rather than to precision.
    """
    if not enabled or device.type != "cuda":
        return False, None

    if dtype_spec == "auto":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        logger.warning(
            "bf16 unsupported on this GPU; falling back to fp16 + GradScaler. "
            "Watch for NaN through frequency_branch's log(|fft|+1e-8)."
        )
        return True, torch.float16
    return True, {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_spec]


def autocast_ctx(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# --------------------------------------------------------------------------- #
# Freeze schedule
# --------------------------------------------------------------------------- #
def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze/unfreeze the spatial backbone.

    **Both halves matter.** ``requires_grad_(False)`` stops the weights being
    updated by the optimizer, but EfficientNet's BatchNorm layers update their
    running mean/var in the *forward* pass whenever the module is in training
    mode -- no gradient required. So a backbone frozen by requires_grad alone
    still drifts, and the pretrained features degrade over exactly the epochs you
    intended to protect them. Calling .eval() on it is what actually freezes it.
    """
    backbone = getattr(model, "spatial", None)
    if backbone is None:
        return
    for p in backbone.parameters():
        p.requires_grad_(not frozen)
    if frozen:
        backbone.eval()  # the half everyone forgets


def build_optimizer(model: nn.Module, cfg, frozen: bool) -> torch.optim.Optimizer:
    """AdamW with discriminative learning rates (spec: AdamW, lr 1e-4, wd 1e-4).

    Two groups: the pretrained backbone needs a far smaller step than the
    randomly-initialised fusion+head, or the first epochs wash out the ImageNet
    features being transferred.

    Weight decay is not applied to norm/bias parameters -- decaying a LayerNorm
    gain toward zero is just a slow way to break the layer.
    """
    backbone_params, head_params, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):  # norms and biases
            no_decay.append(param)
        elif name.startswith("spatial."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    head_lr = cfg.optim.head_lr_frozen if frozen else cfg.optim.lr
    groups: list[dict[str, Any]] = [
        {"params": head_params, "lr": head_lr, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]
    if backbone_params:
        # A discriminative LR only makes sense when there are pretrained features
        # worth protecting. From scratch, it just cripples 89% of the model:
        # measured, the backbone is 10.6M of 12.0M parameters, so at
        # backbone_lr=1e-5 against lr=1e-3 the network effectively cannot learn.
        # This is exactly what made the T34 smoke test fail against a correct loop.
        if not cfg.model.pretrained and cfg.optim.backbone_lr < cfg.optim.lr:
            logger.warning(
                "pretrained=False but backbone_lr (%.1e) < lr (%.1e). The backbone "
                "is randomly initialised, so there are no pretrained features to "
                "protect -- this only stops ~89%% of the model from learning. Set "
                "backbone_lr == lr when training from scratch.",
                cfg.optim.backbone_lr,
                cfg.optim.lr,
            )
        groups.insert(
            0,
            {
                "params": backbone_params,
                "lr": cfg.optim.backbone_lr,
                "weight_decay": cfg.optim.weight_decay,
            },
        )
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=head_lr, weight_decay=cfg.optim.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg, steps_per_epoch: int
) -> Callable[[int], float]:
    """Linear warmup → cosine decay, **stepped per iteration**.

    Not ReduceLROnPlateau: with 15-30 epochs and early-stop patience 5, plateau
    patience would have to be <=2 to react before the run ends -- too coarse to
    be useful, and it adds state that complicates resume.

    Returns a closure that sets the LR for a global step and returns it, rather
    than a torch scheduler, so warmup+cosine stays one readable expression and
    resume is just "call it with the restored step".
    """
    warmup_steps = max(0, cfg.optim.warmup_epochs * steps_per_epoch)
    total_steps = max(1, cfg.train.epochs * steps_per_epoch)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def set_lr(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            scale = (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
            floor = cfg.optim.lr_min
            group["lr"] = floor + (base - floor) * scale
        return optimizer.param_groups[-1]["lr"]

    return set_lr


def build_criterion(cfg, class_weights: torch.Tensor | None = None) -> nn.Module:
    """CrossEntropy with optional class weights + label smoothing.

    Class weighting is how we keep FF++'s 4:1 fake:real ratio without
    downsampling away 75% of the fakes (T16).
    """
    return nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.optim.label_smoothing
    )


def compute_class_weights(labels: Iterable[int], num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Mean-1 normalisation keeps the loss magnitude comparable to the unweighted
    case, so the LR does not silently need retuning when the prior changes.
    """
    counts = np.bincount(np.asarray(list(labels), dtype=int), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Batch plumbing
# --------------------------------------------------------------------------- #
def _unpack(batch) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
    """Accept both dataset contracts.

    DeepfakeDataset yields ``(tensor, label)`` -- the torchvision convention every
    loop unpacks. DeepfakeVideoDataset yields a dict carrying ``manipulation``,
    which the per-method breakdown needs (T19).
    """
    if isinstance(batch, dict):
        return batch["frames"], batch["label"], batch.get("manipulation")
    x, y = batch
    return x, y, None


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg,
    *,
    set_lr: Callable[[int], float],
    scaler: torch.amp.GradScaler | None = None,
    global_step: int = 0,
    epoch: int = 0,
) -> tuple[float, int]:
    """One epoch. → ``(mean_loss, global_step)``."""
    model.train()
    # Re-assert the freeze every epoch: model.train() above just flipped the
    # whole tree, including the backbone we froze.
    if epoch < cfg.optim.freeze_backbone_epochs:
        set_backbone_frozen(model, True)

    amp_on, amp_dtype = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)
    total_loss, n_batches = 0.0, 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        x, y, _ = _unpack(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.train.channels_last and x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)

        with autocast_ctx(device, amp_dtype if amp_on else None):
            logits = model(x)
            loss = criterion(logits, y)

        # Scale so accumulated gradients average rather than sum -- otherwise the
        # effective LR silently multiplies by accum_steps.
        scaled = loss / cfg.train.accum_steps
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (i + 1) % cfg.train.accum_steps == 0:
            if scaler is not None:
                # MUST unscale before clipping: gradients are still multiplied by
                # the scale factor, so clipping here would clip the wrong number.
                scaler.unscale_(optimizer)
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr = set_lr(global_step)
            global_step += 1

            if global_step % cfg.train.log_every == 0:
                logger.info(
                    "  epoch %d step %d loss %.4f lr %.2e",
                    epoch, global_step, loss.item(), lr,
                )

        total_loss += loss.item()
        n_batches += 1

    # Flush a partial accumulation cycle.
    #
    # With 5 batches and accum_steps=2 the loop steps after batches 2 and 4, and
    # batch 5's gradient is left sitting in .grad -- never stepped, and never
    # zeroed, so it silently leaks into the FIRST step of the next epoch, mixed
    # with a different batch. Data is quietly reweighted and nothing reports it.
    # Any dataset whose batch count is not a multiple of accum_steps hits this,
    # which is most of them.
    if n_batches % cfg.train.accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        if cfg.optim.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        set_lr(global_step)
        global_step += 1
        logger.debug(
            "  flushed a partial accumulation cycle (%d batches, accum=%d)",
            n_batches, cfg.train.accum_steps,
        )

    logger.info(
        "  epoch %d train: loss %.4f over %d batches in %.1fs",
        epoch, total_loss / max(1, n_batches), n_batches, time.time() - started,
    )
    return total_loss / max(1, n_batches), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg,
    *,
    threshold: float | None = None,
) -> Metrics:
    """Full evaluation pass → :class:`Metrics`.

    Accumulates probabilities (not logits): ROC needs a monotone score, and
    thresholds chosen on probabilities have to be applied to probabilities.
    """
    model.eval()
    amp_on, amp_dtype = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)

    scores: list[float] = []
    labels: list[int] = []
    manips: list[str] = []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        x, y, manip = _unpack(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.train.channels_last and x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)

        with autocast_ctx(device, amp_dtype if amp_on else None):
            logits = model(x)
            loss = criterion(logits, y)

        # float() before softmax: under bf16/fp16 autocast the logits come back
        # in low precision, and probabilities feed the metrics.
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        scores.extend(probs.cpu().tolist())
        labels.extend(y.cpu().tolist())
        if manip is not None:
            manips.extend(list(manip))
        total_loss += loss.item()
        n_batches += 1

    return compute_metrics(
        labels,
        scores,
        loss=total_loss / max(1, n_batches),
        manipulations=manips or None,
        threshold=threshold,
    )
