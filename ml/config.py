"""Training configuration for SEETHRU (BUILD_PLAN T25).

**Frozen dataclasses + a YAML overlay.** Not Hydra: its multirun/sweep machinery
and output-dir hijacking buy nothing for the two runs this project makes, and
dataclasses give typed defaults, IDE completion, and an ``asdict()`` that drops
straight into a checkpoint (T24) with no conversion step.

Frozen because a config that mutates mid-run is a config the checkpoint lies
about. If ``epochs`` can be reassigned at epoch 12, the value recorded at save
time is not the value that produced the weights.

**Every number from the project spec lives here and nowhere else.** The spec's
"dropout 0.3-0.5" was previously unreachable from anywhere (hardcoded at 0.4
inside the fusion MLP, T22) which is exactly the failure this file prevents: a
parameter you cannot set is a parameter you cannot tune, and you will not notice
because nothing errors.

Usage::

    cfg = Config.load("ml/configs/image.yaml")
    cfg = Config.load("ml/configs/image.yaml", overrides={"train": {"epochs": 5}})
    checkpoint.save(..., config=cfg.to_checkpoint_dict())
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Constants shared with the data/model code. Defined here so the config is the
# single source of truth, and imported by nothing at runtime -- these are values,
# not behaviour.
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("real", "fake")


@dataclass(frozen=True)
class DataCfg:
    """Dataset location, splitting and loading.

    Spec: image size 224, validation split 15-20%.
    """

    root: str = "data/dummy/images"
    # (train, val, test). 0.15 val sits inside the spec's 15-20%.
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    image_size: int = 224

    # False by default, deliberately (T16). Balancing by downsampling throws away
    # 75% of FF++'s fakes (4 manipulations vs 1 real per identity). Use
    # class-weighted loss instead -- it keeps the data and costs nothing.
    balance: bool = False
    class_weighted_loss: bool = True

    # Windows uses spawn, so each worker re-imports the world. 4 is a reasonable
    # default; ml/train.py MUST guard __main__ or workers fork-bomb (T32).
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    # Kept for BatchNorm-era safety; LayerNorm (T21) makes it optional now, so it
    # is a genuine choice rather than a workaround.
    drop_last: bool = True

    # Video only.
    n_frames: int = 16  # spec: frames per video 16


@dataclass(frozen=True)
class ModelCfg:
    """Architecture.

    Spec: dropout 0.3-0.5, EfficientNet-B3 spatial backbone, BiLSTM temporal.
    """

    arch: str = "image"  # "image" | "video"
    # concat, per docs/adr/0001-fusion-mode.md. Branch attribution comes from
    # ablation (T51), which works on any fusion -- so there is no explainability
    # reason to pay attention fusion's messier two-stage transfer.
    fusion: str = "concat"
    dropout: float = 0.4  # spec: 0.3-0.5
    pretrained: bool = True  # ImageNet init for the spatial backbone
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.arch not in ("image", "video"):
            raise ValueError(f"arch must be image|video, got {self.arch!r}")
        if self.fusion not in ("concat", "attention"):
            raise ValueError(f"fusion must be concat|attention, got {self.fusion!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class OptimCfg:
    """Optimizer and schedule.

    Spec: AdamW, lr 1e-4, weight decay 1e-4.
    """

    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Discriminative LRs for the freeze/unfreeze schedule (T30). The pretrained
    # backbone needs a far smaller step than the randomly-initialised head, or
    # the first epochs wash out the ImageNet features you are transferring.
    backbone_lr: float = 1e-5
    head_lr_frozen: float = 1e-3  # while the backbone is frozen

    # 1 epoch linear warmup -> cosine to lr_min, stepped per-iteration.
    # Not ReduceLROnPlateau: with 15-30 epochs and early-stop patience 5, its
    # patience would have to be <=2 to react at all, and it complicates resume.
    warmup_epochs: int = 1
    lr_min: float = 1e-6
    clip_grad_norm: float = 1.0

    # Epochs to keep the spatial backbone frozen. NOTE (T30): freezing must also
    # call .eval() on it -- requires_grad=False does NOT stop EfficientNet's
    # BatchNorms updating their running stats, so a "frozen" backbone silently
    # drifts.
    freeze_backbone_epochs: int = 2
    label_smoothing: float = 0.05  # predict() surfaces confidence; do not overcook it


@dataclass(frozen=True)
class TrainCfg:
    """Loop control.

    Spec: batch size 8-16, epochs 15-30, early stopping patience 5 on val loss,
    restore best weights.
    """

    epochs: int = 20  # spec: 15-30
    batch_size: int = 16  # spec: 8-16
    # Effective batch = batch_size * accum_steps = 32.
    accum_steps: int = 2

    amp: bool = True
    # bf16 when the GPU supports it, else fp16+GradScaler. bf16 specifically
    # because frequency_branch's log(|fft|+1e-8) is an exponent-range hazard and
    # bf16 needs no loss scaling. Resolved at runtime; "auto" here.
    amp_dtype: str = "auto"  # "auto" | "bf16" | "fp16"

    patience: int = 5  # spec
    monitor: str = "val_loss"  # spec
    restore_best: bool = True  # spec

    seed: int = 42
    # Off by default: cuDNN's BiLSTM backward has NO deterministic kernel and
    # raises under use_deterministic_algorithms(True). Enable for smoke runs only.
    deterministic: bool = False

    # Off by default: Triton on Windows is fragile and the cuDNN LSTM graph-breaks.
    # Keep the flag; measure before trusting it.
    compile: bool = False
    channels_last: bool = True

    device: str = "auto"  # "auto" | "cpu" | "cuda"
    out_dir: str = "runs"
    name: str = "image"

    log_every: int = 20  # iterations

    def __post_init__(self) -> None:
        if self.monitor not in ("val_loss", "val_auc"):
            raise ValueError(f"monitor must be val_loss|val_auc, got {self.monitor!r}")
        if self.amp_dtype not in ("auto", "bf16", "fp16"):
            raise ValueError(f"amp_dtype must be auto|bf16|fp16, got {self.amp_dtype!r}")
        if self.accum_steps < 1:
            raise ValueError(f"accum_steps must be >= 1, got {self.accum_steps}")


@dataclass(frozen=True)
class Config:
    """The whole configuration for one run."""

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        """Build from a nested dict, rejecting unknown keys.

        Unknown keys are an ERROR, not a warning. A typo'd ``droput: 0.5`` that
        is silently ignored means you run the whole experiment at 0.4 and compare
        it against another run at 0.4, concluding dropout has no effect.
        """
        sections = {
            "data": DataCfg,
            "model": ModelCfg,
            "optim": OptimCfg,
            "train": TrainCfg,
        }
        unknown_sections = set(raw) - set(sections)
        if unknown_sections:
            raise ValueError(
                f"unknown config section(s): {sorted(unknown_sections)}. "
                f"Expected: {sorted(sections)}"
            )

        built: dict[str, Any] = {}
        for name, klass in sections.items():
            values = raw.get(name) or {}
            if not isinstance(values, dict):
                raise ValueError(f"config section {name!r} must be a mapping")
            valid = {f.name for f in dataclasses.fields(klass)}
            unknown = set(values) - valid
            if unknown:
                raise ValueError(
                    f"unknown key(s) in [{name}]: {sorted(unknown)}. "
                    f"Valid keys: {sorted(valid)}"
                )
            # YAML gives lists where the dataclass wants tuples.
            coerced = {
                k: tuple(v) if isinstance(v, list) else v for k, v in values.items()
            }
            built[name] = klass(**coerced)
        return cls(**built)

    @classmethod
    def load(
        cls, path: str | Path, overrides: dict[str, Any] | None = None
    ) -> Config:
        """Load a YAML file, then apply ``overrides`` (same nested shape)."""
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if overrides:
            raw = _deep_merge(raw, overrides)
        return cls.from_dict(raw)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Flat, primitive-only dict for the checkpoint (T24).

        Flat and flattened on purpose: ``load_for_inference`` reads ``arch``,
        ``fusion``, ``image_size`` etc. directly, and must not need to know this
        module's nesting. The backend imports zero training code, so it cannot
        import ``Config`` to interpret its own checkpoint.

        Everything here is a primitive -- ``torch.load(weights_only=True)``, the
        default since torch 2.6, cannot unpickle anything else.
        """
        return {
            "arch": self.model.arch,
            "fusion": self.model.fusion,
            "num_classes": self.model.num_classes,
            "dropout": self.model.dropout,
            "image_size": self.data.image_size,
            "n_frames": self.data.n_frames,
            "class_names": list(CLASS_NAMES),
            "norm_mean": list(IMAGENET_MEAN),
            "norm_std": list(IMAGENET_STD),
            # False unless T78's calibration step explicitly flips it. The API
            # must not present a raw softmax as a probability (T58).
            "calibrated": False,
            # Full nested config for provenance/debugging.
            "full": self.to_dict(),
        }

    def replace(self, **section_overrides: dict[str, Any]) -> Config:
        """Return a new Config with sections updated (frozen, so never in place)."""
        return Config.from_dict(_deep_merge(self.to_dict(), section_overrides))


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
