"""Training entry point for SEETHRU (BUILD_PLAN T32).

    python ml/train.py --config ml/configs/smoke.yaml --smoke     # 2 min, CPU
    python ml/train.py --config ml/configs/image.yaml             # stage 1
    python ml/train.py --config ml/configs/video.yaml --init-from runs/image/best.pt
    python ml/train.py --config ml/configs/image.yaml --resume runs/image/last.pt

**The __main__ guard at the bottom is not a formality.** Windows spawns
DataLoader workers by re-importing __main__; without the guard each worker
re-executes this module top to bottom and forks its own workers, recursively.
It is a fork bomb, and it is the default platform here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.manifest import MANIFEST_NAME  # noqa: E402
from ml import checkpoint  # noqa: E402
from ml.config import Config  # noqa: E402
from ml.engine import (  # noqa: E402
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
from ml.utils.logging import log_run_header, setup_logging  # noqa: E402
from ml.utils.seed import make_generator, seed_everything, worker_init_fn  # noqa: E402
from ml.utils.tracking import Tracker  # noqa: E402

logger = logging.getLogger("seethru.train")

SMOKE_MAX_LOSS = 0.05
SMOKE_MIN_ACC = 1.0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loaders(cfg: Config, smoke: bool = False):
    """Build train/val loaders. → ``(train_loader, val_loader, class_weights)``.

    Two data paths, chosen by what is actually on disk:

    * ``<root>/manifest.jsonl`` present -> the **processed** datasets (T41b).
      This is the real path: one face-detection pass produced ``.npy`` + a
      manifest, and stage 1 reads the frame view of it while stage 2 reads the
      clip view.
    * otherwise -> ``DeepfakeDataset`` over ``real/``+``fake/`` image folders,
      which is what the dummy set provides and what the smoke test runs on.

    Auto-detecting beats a config flag: the flag would be one more thing to set
    wrong, and the answer is unambiguous from the filesystem.
    """
    from ml.preprocessing.augmentation import build_train_transform, build_val_transform

    val_tf = build_val_transform(cfg.data.image_size)
    # A smoke run uses the VAL transform for training. This is essential, not
    # tidiness: train augmentation (RandomCrop, Rotate, GaussNoise, blur, JPEG)
    # makes 10 samples genuinely unmemorisable -- every epoch is a different
    # image -- so the overfit assert would fail against a perfectly correct loop.
    train_tf = val_tf if smoke else build_train_transform(cfg.data.image_size)

    root = Path(cfg.data.root)
    if (root / MANIFEST_NAME).is_file():
        train_ds, val_ds, labels = _build_processed(cfg, root, train_tf, val_tf)
    else:
        if cfg.model.arch != "image":
            raise NotImplementedError(
                f"video training needs the processed manifest from T41/T69, but "
                f"{root / MANIFEST_NAME} does not exist. Run:\n"
                f"  python ml/preprocessing/prepare_datasets.py --ff_root ... --out {root}"
            )
        train_ds, val_ds, labels = _build_image_folders(cfg, train_tf, val_tf)

    if smoke:
        train_ds = _take_balanced(train_ds, n_per_class=5)
        val_ds = train_ds  # memorisation check: val IS train, deliberately

    logger.info("train: %d samples | val: %d samples", len(train_ds), len(val_ds))

    class_weights = None
    if cfg.data.class_weighted_loss and not smoke:
        class_weights = compute_class_weights(labels)
        logger.info("class weights: %s", class_weights.tolist())

    loader_kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        worker_init_fn=worker_init_fn,
        generator=make_generator(cfg.train.seed),
    )
    if cfg.data.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.data.persistent_workers

    # shuffle=False for smoke, deliberately. With BatchNorm in the backbone, a
    # sample's output depends on which other samples share its batch -- so
    # reshuffling 10 samples into different batches of 5 each epoch makes the
    # train loss bounce on batch *composition* rather than on the weights.
    # Measured: the loss swung 0.0045 -> 0.89 between epochs at lr=1e-6, where
    # the weights are effectively frozen and real divergence is impossible.
    # Shuffling buys nothing on 10 samples and costs a readable signal.
    train_loader = DataLoader(
        train_ds, shuffle=not smoke, drop_last=cfg.data.drop_last, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, class_weights


def _build_processed(cfg: Config, root: Path, train_tf, val_tf):
    """The real path: frame view (stage 1) / clip view (stage 2) over one manifest."""
    from data.processed_dataset import DeepfakeClipDataset, DeepfakeFrameDataset

    klass = DeepfakeFrameDataset if cfg.model.arch == "image" else DeepfakeClipDataset
    logger.info("Using the processed manifest at %s (%s)", root, klass.__name__)

    # Splits come from the manifest, which got them from FF++'s official
    # splits/*.json (T15) -- NOT re-derived here. Re-deriving would reintroduce
    # the two-identity leak the loaders exist to prevent.
    train_ds = klass(root, split="train", dataset="ffpp", transform=train_tf)
    val_ds = klass(root, split="val", dataset="ffpp", transform=val_tf)
    return train_ds, val_ds, train_ds.labels()


def _build_image_folders(cfg: Config, train_tf, val_tf):
    """The dummy/smoke path: real/ + fake/ image folders."""
    from data.dataset_manager import DeepfakeDataset

    common = dict(
        root=cfg.data.root,
        split_ratios=cfg.data.split_ratios,
        balance=cfg.data.balance,
        image_size=cfg.data.image_size,
    )
    train_ds = DeepfakeDataset(**common, split="train", transform=train_tf)
    val_ds = DeepfakeDataset(**common, split="val", transform=val_tf)
    return train_ds, val_ds, [label for _p, label, _i in train_ds.samples]


def _take_balanced(dataset, n_per_class: int):
    """A tiny balanced subset for the smoke test.

    Works over either data path: the processed datasets expose ``labels()``, the
    image-folder one exposes ``samples``.
    """
    from torch.utils.data import Subset

    if hasattr(dataset, "labels"):
        labels = dataset.labels()
    else:
        labels = [label for _p, label, _i in dataset.samples]

    picked: dict[int, list[int]] = {0: [], 1: []}
    for i, label in enumerate(labels):
        if len(picked[label]) < n_per_class:
            picked[label].append(i)
    indices = sorted(picked[0] + picked[1])
    if len(indices) < 2 * n_per_class:
        raise RuntimeError(
            f"smoke needs {n_per_class} samples per class, found "
            f"{ {k: len(v) for k, v in picked.items()} }"
        )
    return Subset(dataset, indices)


def build_model(cfg: Config):
    from ml.models.classifier import ImageClassifier, VideoClassifier

    klass = {"image": ImageClassifier, "video": VideoClassifier}[cfg.model.arch]
    return klass(
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
        fusion=cfg.model.fusion,
    )


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(cfg: Config, smoke: bool = False, init_from: str | None = None,
          resume: str | None = None) -> dict:
    run_dir = Path(cfg.train.out_dir) / cfg.train.name
    setup_logging(run_dir)
    log_run_header(cfg, run_dir)

    seed_everything(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    logger.info("device: %s", device)
    if device.type == "cpu" and not smoke:
        logger.warning(
            "Training on CPU. Fine for a smoke run; a real run will take days. "
            "Install a CUDA build of torch (see requirements.txt) -- T72."
        )

    train_loader, val_loader, class_weights = build_loaders(cfg, smoke=smoke)
    model = build_model(cfg).to(device)
    if cfg.train.channels_last and cfg.model.arch == "image":
        model = model.to(memory_format=torch.channels_last)

    if init_from:
        # Stage 2: load stage 1's spatial/frequency/fusion/classifier. RAISES
        # unless exactly temporal.* is missing -- otherwise a renamed module
        # means we silently train from scratch while reporting a transfer (T33).
        payload = checkpoint.load_raw(init_from, map_location=str(device))
        missing, _ = checkpoint.transfer_image_to_video(model, payload["model"])
        logger.info("initialised from %s (new: %d temporal params)", init_from, len(missing))

    frozen_epochs = cfg.optim.freeze_backbone_epochs
    set_backbone_frozen(model, frozen_epochs > 0)
    optimizer = build_optimizer(model, cfg, frozen=frozen_epochs > 0)
    steps_per_epoch = max(1, len(train_loader) // cfg.train.accum_steps)
    set_lr = build_scheduler(optimizer, cfg, steps_per_epoch)

    amp_on, amp_dtype = resolve_amp(cfg.train.amp, cfg.train.amp_dtype, device)
    # A GradScaler is only needed for fp16; bf16 has fp32's exponent range.
    scaler = (
        torch.amp.GradScaler(device.type)
        if amp_on and amp_dtype is torch.float16
        else None
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = build_criterion(cfg, class_weights)

    start_epoch, global_step, best = 0, 0, float("inf")
    if resume:
        payload = checkpoint.load_for_training(
            resume, model, optimizer, scaler=scaler, map_location=str(device)
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        logger.info("resumed from %s at epoch %d", resume, start_epoch)

    best_state, epochs_without_improvement = None, 0
    history: list[dict] = []

    with Tracker(run_dir) as tracker:
        for epoch in range(start_epoch, cfg.train.epochs):
            # Unfreeze exactly once, and rebuild the optimizer so the backbone's
            # params enter with their own (much smaller) LR.
            if epoch == frozen_epochs and frozen_epochs > 0:
                logger.info("unfreezing backbone at epoch %d", epoch)
                set_backbone_frozen(model, False)
                optimizer = build_optimizer(model, cfg, frozen=False)
                set_lr = build_scheduler(optimizer, cfg, steps_per_epoch)

            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, device, cfg,
                set_lr=set_lr, scaler=scaler, global_step=global_step, epoch=epoch,
            )
            metrics = evaluate(model, val_loader, criterion, device, cfg)
            logger.info("  epoch %d val: %s", epoch, metrics.summary())

            tracker.scalar("train/loss", train_loss, epoch)
            tracker.metrics("val", metrics, epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics.to_dict()})

            monitored = metrics.loss if cfg.train.monitor == "val_loss" else -metrics.auc
            improved = monitored < best
            if improved:
                best = monitored
                epochs_without_improvement = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                checkpoint.save(
                    run_dir / "best.pt", model, epoch=epoch, global_step=global_step,
                    config=cfg.to_checkpoint_dict(), optimizer=optimizer, scaler=scaler,
                    metrics=metrics.to_dict(),
                )
                logger.info("  new best (%s=%.4f) -> best.pt", cfg.train.monitor, monitored)
            else:
                epochs_without_improvement += 1

            checkpoint.save(
                run_dir / "last.pt", model, epoch=epoch, global_step=global_step,
                config=cfg.to_checkpoint_dict(), optimizer=optimizer, scaler=scaler,
                metrics=metrics.to_dict(),
            )

            if epochs_without_improvement >= cfg.train.patience:
                logger.info(
                    "early stopping: no improvement in %d epochs (spec: patience %d)",
                    epochs_without_improvement, cfg.train.patience,
                )
                break

    # Spec: restore best weights before the final pass.
    if cfg.train.restore_best and best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored best weights")

    final = evaluate(model, val_loader, criterion, device, cfg)
    logger.info("final val: %s", final.summary())

    # The operating point is chosen HERE, on val, and frozen. select_threshold
    # refuses any other split -- tuning on test is invisible in the output.
    if not smoke:
        logger.info("val EER threshold (frozen for test/cross-dataset): %.4f", final.eer_threshold)

    if smoke:
        _assert_smoke_passed(history, final)

    return {"history": history, "final": final.to_dict(), "run_dir": str(run_dir)}


def _assert_smoke_passed(history: list[dict], final) -> None:
    """The smoke contract (T34): if it cannot memorise 10 samples, it is broken.

    This is the whole point of the smoke run. A loop with a detached graph, a
    zeroed LR, a mis-scaled accumulation, or inverted labels will train
    "successfully" for hours and produce nothing -- and on a shared GPU you find
    out days later. Ten samples on a CPU answers it in two minutes.

    **It asserts on TRAIN-mode loss, deliberately, and NOT on eval accuracy.**
    That is not laziness; eval accuracy is unmeasurable here and asserting on it
    made this test fail against a working loop. EfficientNet's BatchNorms
    estimate running statistics from the batches they see, and 10 samples over 50
    steps is nowhere near enough for those estimates to converge. Measured: the
    model reached train loss 0.02 while eval mode assigned every one of the 10
    samples an identical p_fake of 0.996 -- perfectly memorised, and useless in
    eval, because eval used garbage running stats.

    "Do gradients flow and reduce the loss?" is a train-mode question. BatchNorm
    convergence is a different question that genuinely needs real data, and
    conflating them means the smoke test reports a broken loop when the loop is
    fine -- which is worse than no smoke test, because you go and look for a bug
    that is not there.
    """
    if not history:
        raise RuntimeError("SMOKE FAILED: no epochs ran")

    losses = [h["train_loss"] for h in history]
    best_loss = min(losses)
    logger.info("=" * 70)
    logger.info("SMOKE CHECK  (train-mode loss; see docstring re: eval/BatchNorm)")
    logger.info(
        "  train loss: %.4f -> %.4f over %d epochs (best %.4f)",
        losses[0], losses[-1], len(losses), best_loss,
    )
    logger.info("  val AUC (ranking, BN-independent): %.4f", final.auc)

    failures = []
    # min(), not the last epoch. The single question this test asks is "CAN the
    # loop drive the loss to zero on data it has seen?" -- if it reached 0.005 at
    # any point then gradients flow, the LR is non-zero, the labels are not
    # shuffled, and the accumulation scaling is right. That is the whole contract.
    #
    # There is deliberately NO "did the loss end low" or "did it go up" check.
    # An earlier version had one and it failed a working loop: with BatchNorm in
    # the backbone, the train loss depends on batch *composition*, so it swings
    # (measured: 0.0045 -> 0.89 at lr=1e-6, where the weights cannot move at all).
    # shuffle=False now removes most of that, but the tail of a converged
    # 10-sample run is still not a signal worth gating on.
    if best_loss > SMOKE_MAX_LOSS:
        failures.append(
            f"best train loss {best_loss:.4f} > {SMOKE_MAX_LOSS} -- the model never "
            f"memorised 10 samples. The loop is broken: check that "
            f"loss.backward() reaches the params, that the LR is non-zero, that "
            f"labels are not shuffled, and that augmentation is off. Also check "
            f"backbone_lr: if pretrained=False and backbone_lr << lr, 89% of the "
            f"model is frozen in all but name."
        )
    if failures:
        for f in failures:
            logger.error("  FAIL %s", f)
        raise RuntimeError("SMOKE FAILED:\n  " + "\n  ".join(failures))

    logger.info("  PASS -- the training loop works. Safe to spend GPU time.")
    logger.info("=" * 70)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SEETHRU model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Overfit ~10 samples and assert the loss collapses (T34).",
    )
    parser.add_argument("--init-from", type=str, help="Stage-1 checkpoint (T33).")
    parser.add_argument("--resume", type=str, help="Resume from last.pt.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    parser.add_argument("--device", type=str, help="Override train.device.")
    args = parser.parse_args()

    overrides: dict = {"train": {}}
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.device is not None:
        overrides["train"]["device"] = args.device

    cfg = Config.load(args.config, overrides=overrides if overrides["train"] else None)
    train(cfg, smoke=args.smoke, init_from=args.init_from, resume=args.resume)


if __name__ == "__main__":
    # REQUIRED on Windows: DataLoader workers spawn by re-importing __main__.
    # Without this guard each worker re-runs the module and spawns its own.
    main()
