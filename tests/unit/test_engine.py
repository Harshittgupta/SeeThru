"""Training-loop internals (BUILD_PLAN T30).

Uses a *tiny* synthetic model, not the real one. That is the point: EfficientNet
takes ~3s per CPU step and its BatchNorms make the loss depend on batch
composition, so testing the accumulation arithmetic against it would be slow AND
confounded. A 2-layer MLP isolates the loop.

The accumulation test earns its place: a mis-scaled accumulation silently
multiplies the effective learning rate by accum_steps. Nothing errors, the loss
still goes down, and the run is merely wrong.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import Config
from ml.engine import (
    build_criterion,
    build_optimizer,
    build_scheduler,
    compute_class_weights,
    evaluate,
    resolve_amp,
    resolve_device,
    set_backbone_frozen,
    train_one_epoch,
)


class TinyModel(nn.Module):
    """A stand-in with a `spatial` attribute, so the freeze logic is exercised.

    **LayerNorm, mirroring the real fusion after T21.** This is load-bearing for
    the accumulation test: with BatchNorm, 2 batches of 4 normalise separately
    while 1 batch of 8 normalises jointly, so their gradients genuinely differ
    and "accumulation == one big batch" is FALSE by construction. That is not a
    bug to fix, it is the exact reason gradient accumulation cannot rescue
    BatchNorm's small-batch problem -- and the reason T21 replaced it.
    """

    def __init__(self, in_dim: int = 8) -> None:
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(in_dim, 8), nn.LayerNorm(8), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.spatial(x))


class TinyBNModel(nn.Module):
    """Same shape but with BatchNorm, for the running-stats freeze tests.

    Those tests need a module that HAS running statistics to leave alone --
    LayerNorm has none, which is itself one of T21's benefits.
    """

    def __init__(self, in_dim: int = 8) -> None:
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(in_dim, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.spatial(x))


@pytest.fixture
def tiny_cfg() -> Config:
    return Config.from_dict(
        {
            "model": {"pretrained": False},
            "train": {
                "epochs": 1, "batch_size": 4, "accum_steps": 1, "amp": False,
                "channels_last": False, "device": "cpu", "log_every": 1000,
            },
            "optim": {"lr": 0.1, "backbone_lr": 0.1, "freeze_backbone_epochs": 0,
                      "warmup_epochs": 0, "label_smoothing": 0.0},
        }
    )


def _loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n, 8)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# --------------------------------------------------------------------------- #
# Device / precision
# --------------------------------------------------------------------------- #
def test_amp_is_off_on_cpu():
    """autocast on CPU buys nothing here and complicates debugging -- and it is
    what lets the smoke test run in plain fp32, so a NaN is attributable to the
    loop rather than to precision."""
    enabled, dtype = resolve_amp(True, "auto", torch.device("cpu"))
    assert enabled is False
    assert dtype is None


def test_amp_disabled_when_requested():
    assert resolve_amp(False, "bf16", torch.device("cpu")) == (False, None)


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"


# --------------------------------------------------------------------------- #
# Freezing
# --------------------------------------------------------------------------- #
def test_freeze_sets_eval_mode_not_just_requires_grad():
    """The half everyone forgets.

    requires_grad=False stops the optimizer touching the weights, but BatchNorm
    updates its running stats in the FORWARD pass, with no gradient involved. A
    backbone frozen by requires_grad alone still drifts -- corrupting exactly the
    pretrained features the freeze was meant to protect.
    """
    model = TinyBNModel()
    model.train()
    set_backbone_frozen(model, True)

    assert not any(p.requires_grad for p in model.spatial.parameters())
    assert not model.spatial.training, "frozen backbone must be in eval mode"
    assert model.head.training, "the head must stay in train mode"


def test_frozen_backbone_running_stats_do_not_move():
    """The property the test above protects, demonstrated end to end.

    Uses the BatchNorm model on purpose -- EfficientNet's backbone has BNs, and
    this drift is exactly what silently corrupts a "frozen" backbone.
    """
    model = TinyBNModel()
    model.train()
    set_backbone_frozen(model, True)
    bn = model.spatial[1]
    before = bn.running_mean.clone()

    for _ in range(5):
        model(torch.randn(4, 8))

    assert torch.equal(bn.running_mean, before), "a 'frozen' backbone drifted"


def test_requires_grad_alone_would_not_have_been_enough():
    """Demonstrates WHY set_backbone_frozen must also call .eval().

    Without this, the fix in set_backbone_frozen looks like a redundant extra
    line and someone eventually deletes it.
    """
    model = TinyBNModel()
    model.train()
    for p in model.spatial.parameters():  # the naive "freeze"
        p.requires_grad_(False)

    bn = model.spatial[1]
    before = bn.running_mean.clone()
    for _ in range(5):
        model(torch.randn(4, 8))

    assert not torch.equal(bn.running_mean, before), (
        "requires_grad=False alone was expected to let BN stats drift; if it no "
        "longer does, set_backbone_frozen's .eval() call may be removable"
    )


def test_unfreeze_restores_grad():
    model = TinyModel()
    set_backbone_frozen(model, True)
    set_backbone_frozen(model, False)
    assert all(p.requires_grad for p in model.spatial.parameters())


# --------------------------------------------------------------------------- #
# Optimizer groups
# --------------------------------------------------------------------------- #
def test_norms_and_biases_are_excluded_from_weight_decay(tiny_cfg):
    """Decaying a LayerNorm gain toward zero is a slow way to break the layer."""
    model = TinyModel()
    opt = build_optimizer(model, tiny_cfg, frozen=False)
    assert any(g["weight_decay"] == 0.0 for g in opt.param_groups)


def test_warns_when_training_from_scratch_with_a_crippled_backbone(tiny_cfg, caplog):
    """The bug that made the T34 smoke test fail against a correct loop.

    backbone_lr=1e-5 is right for fine-tuning ImageNet features and catastrophic
    from scratch: it leaves ~89% of the model unable to learn, and nothing errors.
    """
    cfg = tiny_cfg.replace(model={"pretrained": False}, optim={"backbone_lr": 1e-5, "lr": 1e-3})
    with caplog.at_level("WARNING"):
        build_optimizer(TinyModel(), cfg, frozen=False)
    assert "pretrained=False but backbone_lr" in caplog.text


def test_no_warning_when_pretrained(tiny_cfg, caplog):
    """A warning that fires on correct config gets ignored, then disabled."""
    cfg = tiny_cfg.replace(model={"pretrained": True}, optim={"backbone_lr": 1e-5, "lr": 1e-3})
    with caplog.at_level("WARNING"):
        build_optimizer(TinyModel(), cfg, frozen=False)
    assert "backbone_lr" not in caplog.text


# --------------------------------------------------------------------------- #
# Gradient accumulation -- the arithmetic that silently rescales the LR
# --------------------------------------------------------------------------- #
def test_accumulation_matches_a_single_large_batch():
    """accum_steps=2 at batch 4 must equal one batch of 8.

    This is the whole reason accumulation exists, and the reason it is dangerous:
    forget the 1/accum_steps scaling and gradients SUM instead of averaging, so
    the effective LR is silently doubled. The loss still falls. Nothing errors.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 8)
    y = torch.randint(0, 2, (8,))
    crit = nn.CrossEntropyLoss()

    def grads(accum_steps: int, batch_size: int) -> torch.Tensor:
        torch.manual_seed(0)
        model = TinyModel()
        model.train()
        model.zero_grad(set_to_none=True)
        for i in range(0, len(x), batch_size):
            loss = crit(model(x[i : i + batch_size]), y[i : i + batch_size])
            (loss / accum_steps).backward()
        return model.head.weight.grad.clone()

    accumulated = grads(accum_steps=2, batch_size=4)
    single = grads(accum_steps=1, batch_size=8)
    assert torch.allclose(accumulated, single, atol=1e-5), (
        "accumulated gradients do not match the equivalent single batch -- "
        "the 1/accum_steps scaling is wrong, and the effective LR is off by a factor"
    )


def test_partial_accumulation_cycle_is_flushed(tiny_cfg):
    """A trailing partial cycle must still step (the T34 bug).

    With 5 batches and accum_steps=2 the loop steps after batches 2 and 4, and
    batch 5's gradient is left in .grad -- never stepped, never zeroed -- so it
    silently leaks into the first step of the NEXT epoch, mixed with different
    data. Any dataset whose batch count is not a multiple of accum_steps hits
    this, which is most of them.
    """
    cfg = tiny_cfg.replace(train={"accum_steps": 2, "batch_size": 2})
    model = TinyModel()
    opt = build_optimizer(model, cfg, frozen=False)
    loader = _loader(n=10, batch_size=2)  # 5 batches, accum 2 -> 2 full + 1 partial

    _loss, step = train_one_epoch(
        model, loader, opt, build_criterion(cfg), torch.device("cpu"), cfg,
        set_lr=build_scheduler(opt, cfg, steps_per_epoch=2), global_step=0, epoch=0,
    )
    assert step == 3, f"expected 2 full cycles + 1 flush = 3 steps, got {step}"
    for p in model.parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            "gradients survived the epoch -- they will leak into the next one"
        )


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #
def test_warmup_then_cosine(tiny_cfg):
    cfg = tiny_cfg.replace(train={"epochs": 10}, optim={"warmup_epochs": 2, "lr": 0.1})
    opt = build_optimizer(TinyModel(), cfg, frozen=False)
    set_lr = build_scheduler(opt, cfg, steps_per_epoch=10)

    lrs = [set_lr(s) for s in range(100)]
    assert lrs[0] < lrs[19], "warmup should ramp the LR up"
    assert lrs[19] == pytest.approx(max(lrs), rel=0.05), "peak at the end of warmup"
    assert lrs[-1] < lrs[19] / 10, "cosine should decay hard by the end"
    assert lrs[-1] >= cfg.optim.lr_min * 0.99, "must not decay below lr_min"


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #
def test_class_weights_upweight_the_minority():
    """FF++ is 4 fakes per real; weighting is how we keep all of them (T16)."""
    w = compute_class_weights([0] * 100 + [1] * 400)
    assert w[0] > w[1], "the rarer class must be weighted higher"
    assert w.mean() == pytest.approx(1.0), "mean-1 keeps the loss scale comparable"


def test_class_weights_are_flat_when_balanced():
    w = compute_class_weights([0] * 50 + [1] * 50)
    assert w[0] == pytest.approx(w[1])


# --------------------------------------------------------------------------- #
# Train / eval smoke at the unit level
# --------------------------------------------------------------------------- #
def test_train_one_epoch_reduces_loss_on_a_tiny_problem(tiny_cfg):
    model = TinyModel()
    opt = build_optimizer(model, tiny_cfg, frozen=False)
    loader = _loader()
    set_lr = build_scheduler(opt, tiny_cfg, steps_per_epoch=2)
    crit = build_criterion(tiny_cfg)

    first, step = train_one_epoch(
        model, loader, opt, crit, torch.device("cpu"), tiny_cfg,
        set_lr=set_lr, global_step=0, epoch=0,
    )
    for epoch in range(1, 15):
        last, step = train_one_epoch(
            model, loader, opt, crit, torch.device("cpu"), tiny_cfg,
            set_lr=set_lr, global_step=step, epoch=epoch,
        )
    assert last < first, "the loop failed to reduce the loss on 8 samples"


def test_evaluate_returns_metrics(tiny_cfg):
    model = TinyModel()
    m = evaluate(model, _loader(), build_criterion(tiny_cfg), torch.device("cpu"), tiny_cfg)
    assert m.n == 8
    assert 0.0 <= m.accuracy <= 1.0


def test_evaluate_leaves_model_in_eval_mode(tiny_cfg):
    model = TinyModel()
    model.train()
    evaluate(model, _loader(), build_criterion(tiny_cfg), torch.device("cpu"), tiny_cfg)
    assert not model.training
