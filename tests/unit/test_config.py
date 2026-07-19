"""Config tests (BUILD_PLAN T25).

The important one is `test_unknown_key_raises`. A typo'd `droput: 0.5` that gets
silently ignored means you run the whole experiment at the default, compare it to
another run at the default, and conclude dropout has no effect. Nothing errors,
nothing looks wrong, and the conclusion is backwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml.config import Config

REPO = Path(__file__).resolve().parents[2]


def test_defaults_match_the_spec():
    """The spec's numbers must be the defaults, not aspirations in a PDF."""
    cfg = Config()
    assert cfg.optim.lr == 0.0001  # spec
    assert cfg.optim.weight_decay == 0.0001  # spec
    assert 8 <= cfg.train.batch_size <= 16  # spec: 8-16
    assert 15 <= cfg.train.epochs <= 30  # spec: 15-30
    assert 0.3 <= cfg.model.dropout <= 0.5  # spec: 0.3-0.5
    assert cfg.data.image_size == 224  # spec
    assert cfg.train.patience == 5  # spec
    assert cfg.train.monitor == "val_loss"  # spec
    assert cfg.train.restore_best is True  # spec
    assert cfg.data.n_frames == 16  # spec
    assert 0.15 <= cfg.data.split_ratios[1] <= 0.20  # spec: val split 15-20%


def test_default_fusion_matches_adr_0001():
    assert Config().model.fusion == "concat"


def test_config_is_frozen():
    """A config that mutates mid-run is one the checkpoint lies about.

    If epochs can be reassigned at epoch 12, the value recorded at save time is
    not the value that produced the weights.
    """
    import dataclasses

    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train.epochs = 99


# --------------------------------------------------------------------------- #
# Typo protection -- the point of this file
# --------------------------------------------------------------------------- #
def test_unknown_key_raises():
    with pytest.raises(ValueError, match=r"unknown key\(s\) in \[model\]"):
        Config.from_dict({"model": {"droput": 0.5}})  # typo


def test_unknown_key_error_lists_valid_keys():
    """The error must be actionable: name the typo AND the alternatives."""
    with pytest.raises(ValueError) as exc:
        Config.from_dict({"optim": {"learning_rate": 1e-3}})  # wrong name
    msg = str(exc.value)
    assert "learning_rate" in msg
    assert "lr" in msg  # the key they meant


def test_unknown_section_raises():
    with pytest.raises(ValueError, match=r"unknown config section"):
        Config.from_dict({"trainer": {"epochs": 3}})


@pytest.mark.parametrize(
    ("section", "values", "match"),
    [
        ("model", {"arch": "audio"}, r"arch must be"),
        ("model", {"fusion": "bogus"}, r"fusion must be"),
        ("model", {"dropout": 1.5}, r"dropout must be"),
        ("train", {"monitor": "val_acc"}, r"monitor must be"),
        ("train", {"amp_dtype": "fp8"}, r"amp_dtype must be"),
        ("train", {"accum_steps": 0}, r"accum_steps must be"),
    ],
)
def test_invalid_values_raise(section, values, match):
    with pytest.raises(ValueError, match=match):
        Config.from_dict({section: values})


# --------------------------------------------------------------------------- #
# YAML round-trip
# --------------------------------------------------------------------------- #
def test_yaml_lists_become_tuples(tmp_path: Path):
    """YAML has no tuples; the dataclass wants one for split_ratios."""
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"data": {"split_ratios": [0.8, 0.1, 0.1]}}))
    cfg = Config.load(path)
    assert cfg.data.split_ratios == (0.8, 0.1, 0.1)


def test_yaml_round_trips(tmp_path: Path):
    cfg = Config.from_dict({"train": {"epochs": 7}, "model": {"dropout": 0.31}})
    path = tmp_path / "c.yaml"
    path.write_text(cfg.to_yaml())
    assert Config.load(path) == cfg


def test_overrides_merge_deeply(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"train": {"epochs": 20, "seed": 1}}))
    cfg = Config.load(path, overrides={"train": {"epochs": 3}})
    assert cfg.train.epochs == 3
    assert cfg.train.seed == 1  # untouched by the override


def test_replace_returns_a_new_config():
    cfg = Config()
    other = cfg.replace(train={"epochs": 99})
    assert other.train.epochs == 99
    assert cfg.train.epochs != 99  # original untouched


# --------------------------------------------------------------------------- #
# Checkpoint interop (T24)
# --------------------------------------------------------------------------- #
def test_checkpoint_dict_is_weights_only_safe(tmp_path: Path, image_model):
    """The config must survive torch.load(weights_only=True) -- the default
    since torch 2.6. A dataclass in the payload poisons the checkpoint forever."""
    import torch

    from ml.checkpoint import save

    cfg = Config()
    path = save(tmp_path / "c.pt", image_model, epoch=0, config=cfg.to_checkpoint_dict())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["config"]["fusion"] == "concat"
    assert payload["config"]["image_size"] == 224


def test_checkpoint_dict_is_flat_for_the_backend():
    """load_for_inference reads these directly and must not know our nesting --
    the backend imports zero training code, so it cannot import Config."""
    d = Config().to_checkpoint_dict()
    for key in ("arch", "fusion", "num_classes", "dropout", "image_size",
                "class_names", "norm_mean", "norm_std", "calibrated"):
        assert key in d, f"backend needs {key!r} at the top level"


def test_checkpoint_dict_defaults_calibrated_false():
    """No calibration code exists yet (T78); the honest default is False."""
    assert Config().to_checkpoint_dict()["calibrated"] is False


# --------------------------------------------------------------------------- #
# The shipped configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["image", "video", "smoke"])
def test_shipped_configs_load(name: str):
    """A config that doesn't parse is found at run time, on the GPU queue."""
    cfg = Config.load(REPO / "ml" / "configs" / f"{name}.yaml")
    assert cfg.train.name == name


def test_smoke_config_can_actually_overfit():
    """The smoke config's whole job is to memorise 10 samples (T34).

    Every assert here is a way that job silently fails -- each one was found by
    the smoke test failing against a *correct* training loop:

    * augmentation makes 10 samples unmemorisable (handled by --smoke swapping in
      val_transform)
    * dropout fights memorisation
    * label smoothing puts a floor under the loss the run can never get under
    * a frozen backbone cannot learn at all
    * and the expensive one: backbone_lr << lr leaves 88.9% of the model
      (10.6M of 12.0M params) crawling at 1/100th the head's rate. That is right
      for fine-tuning ImageNet features and catastrophic from scratch.
    """
    cfg = Config.load(REPO / "ml" / "configs" / "smoke.yaml")
    assert cfg.model.dropout == 0.0, "dropout fights memorisation"
    assert cfg.optim.label_smoothing == 0.0, "smoothing puts a floor under the loss"
    assert cfg.optim.freeze_backbone_epochs == 0, "a frozen backbone cannot memorise"
    assert cfg.model.pretrained is False, "smoke must not need a weight download"
    assert cfg.train.device == "cpu", "smoke must pass with no GPU at all"
    assert cfg.train.deterministic is True
    assert cfg.optim.backbone_lr == cfg.optim.lr, (
        "pretrained=False means the backbone is random -- there are no features "
        "to protect, and backbone_lr < lr just stops 89% of the model learning"
    )


def test_smoke_config_uses_a_full_batch():
    """Full batch is required, and every alternative was tried and measured (T34).

    batch=2 shuffled     -> gradient is BatchNorm noise over 2 samples
    batch=5 shuffled     -> loss swings on batch composition (0.0045 -> 0.89 at
                            lr=1e-6, where weights physically cannot move)
    batch=5 unshuffled   -> WORSE: samples are path-sorted, so batches become
                            single-class and BN leaks the label. Train loss fell
                            while val AUC dropped 1.00 -> 0.60.
    batch=10 (full)      -> one partition, both classes, no composition variance.

    accum_steps is therefore 1: with one batch there is nothing to accumulate.
    The accumulation path is covered by tests/unit/test_engine.py instead, which
    isolates it properly and runs in milliseconds rather than minutes.
    """
    cfg = Config.load(REPO / "ml" / "configs" / "smoke.yaml")
    assert cfg.train.batch_size == 10, "smoke must use the full 10-sample batch"
    assert cfg.train.accum_steps == 1, "one batch means nothing to accumulate"


def test_video_config_batch_size_is_survivable():
    """VideoClassifier flattens (B,T) into the backbone: B=8,T=16 = 128 images
    per step, measured at ~19.3 GB of activations for the spatial branch alone."""
    cfg = Config.load(REPO / "ml" / "configs" / "video.yaml")
    assert cfg.train.batch_size <= 4, "video batch_size > 4 will OOM a 24 GB card"
    assert cfg.model.arch == "video"


def test_image_and_video_fusion_match():
    """Stage 2 loads stage 1's fusion weights; a mismatch breaks transfer (T33)."""
    img = Config.load(REPO / "ml" / "configs" / "image.yaml")
    vid = Config.load(REPO / "ml" / "configs" / "video.yaml")
    assert img.model.fusion == vid.model.fusion
