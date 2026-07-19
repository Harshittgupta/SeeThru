"""Checkpoint contract tests (BUILD_PLAN T24/T31).

The tests that matter here are the ones about `weights_only=True`. torch >= 2.6
made it the default, so a payload containing a Path or a dataclass produces a
checkpoint that saves fine and then **cannot be loaded, ever, by anyone**. That
failure surfaces after training finishes, when the checkpoint is the only artifact
left -- so it has to be caught at save time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.checkpoint import (
    load_for_inference,
    load_for_training,
    load_raw,
    save,
    sha256,
    transfer_image_to_video,
)
from ml.models.classifier import ImageClassifier, VideoClassifier

CONFIG = {
    "arch": "image",
    "fusion": "concat",
    "num_classes": 2,
    "dropout": 0.4,
    "image_size": 224,
    "class_names": ["real", "fake"],
    "norm_mean": [0.485, 0.456, 0.406],
    "norm_std": [0.229, 0.224, 0.225],
}


@pytest.fixture
def ckpt_path(tmp_path: Path, image_model) -> Path:
    return save(
        tmp_path / "best.pt",
        image_model,
        epoch=3,
        config=CONFIG,
        metrics={"val_auc": 0.87},
        eer_threshold=0.42,
    )


# --------------------------------------------------------------------------- #
# weights_only -- the landmine
# --------------------------------------------------------------------------- #
def test_checkpoint_loads_under_weights_only(ckpt_path: Path):
    """The whole point. torch>=2.6 defaults to weights_only=True."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 3
    assert payload["config"]["fusion"] == "concat"


def test_save_rejects_an_unconvertible_object(tmp_path: Path, image_model):
    """Anything _to_plain cannot handle must fail at SAVE, not at load.

    A poisoned checkpoint pickles happily and then cannot be read by anyone,
    forever -- and you find out after the training run is over, when the file is
    the only artifact left. So the check runs before the bytes hit the disk.
    """

    class Opaque:
        pass

    with pytest.raises(TypeError, match=r"cannot unpickle"):
        save(tmp_path / "bad.pt", image_model, epoch=0, config={**CONFIG, "x": Opaque()})


def test_save_rejects_non_string_dict_keys(tmp_path: Path, image_model):
    """weights_only=True requires str/int keys -- a tuple key is a silent poison."""
    with pytest.raises(TypeError, match=r"requires str/int keys"):
        save(
            tmp_path / "bad.pt",
            image_model,
            epoch=0,
            config={**CONFIG, "nested": {(1, 2): "tuple key"}},
        )


def test_save_rejects_deeply_nested_offenders(tmp_path: Path, image_model):
    """The check must recurse -- a config is a tree, and one bad leaf poisons it."""

    class Opaque:
        pass

    with pytest.raises(TypeError, match=r"cannot unpickle"):
        save(
            tmp_path / "bad.pt",
            image_model,
            epoch=0,
            config={**CONFIG, "a": {"b": [{"c": Opaque()}]}},
        )


def test_error_message_locates_the_offender(tmp_path: Path, image_model):
    """The message must say WHERE, not just that something is wrong."""

    class Opaque:
        pass

    with pytest.raises(TypeError) as exc:
        save(
            tmp_path / "bad.pt",
            image_model,
            epoch=0,
            config={**CONFIG, "a": {"b": [{"c": Opaque()}]}},
        )
    assert "config['a']['b'][0]['c']" in str(exc.value)


def test_save_converts_paths_and_dataclasses_automatically(tmp_path: Path, image_model):
    """_to_plain should handle the common cases rather than just complaining."""
    from dataclasses import dataclass

    @dataclass
    class Optim:
        lr: float = 1e-4
        wd: float = 1e-4

    path = save(
        tmp_path / "ok.pt",
        image_model,
        epoch=0,
        config={**CONFIG, "optim": Optim(), "run_dir": tmp_path},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["config"]["optim"] == {"lr": 1e-4, "wd": 1e-4}
    assert isinstance(payload["config"]["run_dir"], str)


def test_error_message_names_the_fix(tmp_path: Path, image_model):
    """An error that says what's wrong but not what to do costs an hour."""
    with pytest.raises(TypeError) as exc:
        save(tmp_path / "bad.pt", image_model, epoch=0, config={**CONFIG, "p": object()})
    assert "Path -> str" in str(exc.value)
    assert "asdict()" in str(exc.value)


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
def test_load_for_inference_reproduces_logits(ckpt_path: Path, image_model):
    """The point of a checkpoint: identical outputs after a round-trip."""
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        before = image_model.eval()(x)

    restored, meta = load_for_inference(ckpt_path)
    with torch.no_grad():
        after = restored(x)

    assert torch.allclose(before, after, atol=1e-6)
    assert meta["arch"] == "image"
    assert meta["eer_threshold"] == pytest.approx(0.42)


def test_load_for_inference_returns_model_in_eval_mode(ckpt_path: Path):
    """Serving a model in train() mode applies dropout to every prediction."""
    model, _ = load_for_inference(ckpt_path)
    assert not model.training


def test_meta_defaults_calibrated_to_false(ckpt_path: Path):
    """Uncalibrated is the honest default -- no calibration code exists yet (T78).

    If this ever defaulted True, the API would present a raw softmax as a
    probability (T58).
    """
    _, meta = load_for_inference(ckpt_path)
    assert meta["calibrated"] is False


def test_load_for_inference_rebuilds_arch_from_checkpoint(tmp_path: Path):
    """fusion/arch must come from the file, not from a hardcoded default.

    A backend assuming "concat" would silently mis-load an attention checkpoint.
    """
    model = VideoClassifier(pretrained=False, fusion="attention", dropout=0.3)
    path = save(
        tmp_path / "vid.pt",
        model,
        epoch=1,
        config={**CONFIG, "arch": "video", "fusion": "attention", "dropout": 0.3},
    )
    restored, meta = load_for_inference(path)
    assert isinstance(restored, VideoClassifier)
    assert meta["fusion"] == "attention"
    from ml.models.fusion import AttentionFusion

    assert isinstance(restored.fusion, AttentionFusion)


def test_unknown_arch_raises(tmp_path: Path, image_model):
    path = save(tmp_path / "x.pt", image_model, epoch=0, config={**CONFIG, "arch": "bogus"})
    with pytest.raises(ValueError, match=r"arch='bogus'"):
        load_for_inference(path)


# --------------------------------------------------------------------------- #
# Atomicity, provenance, resume
# --------------------------------------------------------------------------- #
def test_save_leaves_no_temp_files(tmp_path: Path, image_model):
    save(tmp_path / "a.pt", image_model, epoch=0, config=CONFIG)
    assert not list(tmp_path.glob("*.tmp")), "atomic save leaked a temp file"


def test_save_overwrites_atomically(tmp_path: Path, image_model):
    """best.pt is rewritten every improvement; a truncated write must be
    impossible, since it is often the only copy of an expensive run."""
    p = save(tmp_path / "best.pt", image_model, epoch=0, config=CONFIG)
    first = sha256(p)
    save(tmp_path / "best.pt", image_model, epoch=9, config=CONFIG)
    assert sha256(p) != first
    assert load_raw(p)["epoch"] == 9


def test_checkpoint_records_provenance(ckpt_path: Path):
    """'Which code made this?' must be answerable from the file alone."""
    payload = load_raw(ckpt_path)
    assert "git" in payload
    assert set(payload["git"]) == {"sha", "dirty"}


def test_resume_restores_optimizer_and_step(tmp_path: Path):
    model = ImageClassifier(pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Take a step so the optimizer has real state to restore.
    model(torch.randn(2, 3, 224, 224)).sum().backward()
    opt.step()

    path = save(
        tmp_path / "last.pt", model, epoch=5, config=CONFIG, optimizer=opt, global_step=123
    )

    model2 = ImageClassifier(pretrained=False)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    payload = load_for_training(path, model2, optimizer=opt2)

    assert payload["epoch"] == 5
    assert payload["global_step"] == 123
    assert opt2.state_dict()["state"], "optimizer state was not restored"


# --------------------------------------------------------------------------- #
# Two-stage transfer (T33)
# --------------------------------------------------------------------------- #
def test_transfer_image_to_video_moves_all_shared_branches():
    img = ImageClassifier(pretrained=False)
    vid = VideoClassifier(pretrained=False)
    missing, unexpected = transfer_image_to_video(vid, img.state_dict())

    assert not unexpected
    assert missing, "expected temporal.* to be missing"
    assert all(k.startswith("temporal.") for k in missing)

    # And the weights genuinely moved.
    for (name, a), (_, b) in zip(
        img.spatial.named_parameters(), vid.spatial.named_parameters(), strict=True
    ):
        assert torch.equal(a, b), f"spatial.{name} did not transfer"


def test_transfer_raises_when_paths_drift():
    """The assertion that stops 'transfer' silently becoming 'from scratch'."""
    img = ImageClassifier(pretrained=False)
    vid = VideoClassifier(pretrained=False)
    sd = {k.replace("fusion.", "fusion_renamed."): v for k, v in img.state_dict().items()}

    with pytest.raises(RuntimeError, match=r"did not line up"):
        transfer_image_to_video(vid, sd)
