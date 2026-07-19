"""Checkpoint save/load for SEETHRU (BUILD_PLAN T24/T31).

Two rules govern this module.

**1. The payload must be `weights_only=True`-loadable.**
torch >= 2.6 flipped ``torch.load``'s default to ``weights_only=True``. Under that
mode the unpickler accepts only a small allowlist of types -- tensors, and plain
containers of primitives. A single ``Path``, dataclass, ``np.dtype`` or custom
class anywhere in the payload makes **every future load raise**, including loads
by code you have not written yet. So the config is stored as a plain dict of
primitives, and :func:`save` asserts it round-trips before returning. Finding
this at save time costs a second; finding it at load time costs the checkpoint.

**2. `load_for_inference` must not import training code.**
The backend (T54) calls it. If it reached into ``ml/train.py`` or a config
dataclass, the API image would need the whole training stack -- and would break
whenever training internals were refactored. It rebuilds the architecture from
primitives stored in the checkpoint, and imports nothing but ``ml.models``.

What lives in a checkpoint, and why:

    model/optimizer/scheduler/scaler   resume-from-crash
    epoch, global_step                 resume, and provenance
    config (plain dict)                rebuild the arch without guessing
    git_sha + git_dirty                which code produced this
    metrics                            what it scored, next to the weights
    rng_state                          exact resume
    class_names, image_size, norm      inference needs these and must not
                                       hardcode them
    eer_threshold                      chosen on val, frozen (T35). Shipping it
                                       here is what stops someone re-tuning it
                                       on test later.
    branch_means                       training-set mean per branch, for ablation
                                       attribution (T51/ADR 0001) -- ablating to
                                       zero would be off-manifold
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1

# Anything outside this set breaks weights_only=True loading.
_PRIMITIVES = (str, int, float, bool, type(None))


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_revision(repo_root: Path | None = None) -> dict:
    """``{"sha": ..., "dirty": ...}`` for the working tree, best-effort.

    Never raises: a checkpoint must still save on a machine without git, or
    outside a repo. But it does **warn**, because "which code produced these
    weights?" is the one question a checkpoint has to be able to answer six weeks
    later, and losing that silently is how it stops being answerable.
    """
    root = str(repo_root or Path(__file__).resolve().parents[1])
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return {"sha": sha, "dirty": bool(status)}
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "git not found (%s) -- this checkpoint will record sha='unknown', so "
            "there will be no way to tell which code produced it. On Windows, add "
            "'C:\\Program Files\\Git\\cmd' to PATH and restart your terminal.",
            exc.__class__.__name__,
        )
    except subprocess.SubprocessError as exc:
        logger.warning(
            "git failed (%s) -- recording sha='unknown'. Is this a git repo?",
            exc.__class__.__name__,
        )
    return {"sha": "unknown", "dirty": False}


# --------------------------------------------------------------------------- #
# weights_only safety
# --------------------------------------------------------------------------- #
def _assert_weights_only_safe(obj: Any, path: str = "config") -> None:
    """Recursively reject anything ``weights_only=True`` cannot unpickle.

    Runs at SAVE time. The alternative is discovering it at load time, when the
    training run is over and the checkpoint is the only artifact left.
    """
    if isinstance(obj, _PRIMITIVES):
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _assert_weights_only_safe(item, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, (str, int)):
                raise TypeError(
                    f"{path}: dict key {key!r} is {type(key).__name__}; "
                    f"weights_only=True requires str/int keys"
                )
            _assert_weights_only_safe(value, f"{path}[{key!r}]")
        return
    raise TypeError(
        f"{path} contains a {type(obj).__name__}, which torch.load"
        f"(weights_only=True) cannot unpickle -- and that is the default from "
        f"torch 2.6. Every future load of this checkpoint would raise.\n"
        f"  Convert it to a primitive first: Path -> str, dataclass -> "
        f"asdict(), tuple of Paths -> list of str, np.float32 -> float."
    )


def _to_plain(obj: Any) -> Any:
    """Best-effort conversion of common offenders into primitives."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_plain(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, _PRIMITIVES):
        return obj
    if hasattr(obj, "item"):  # numpy scalar / 0-d tensor
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            pass
    return obj  # let _assert_weights_only_safe produce the error


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save(
    path: str | Path,
    model: torch.nn.Module,
    *,
    epoch: int,
    config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    global_step: int = 0,
    metrics: dict | None = None,
    eer_threshold: float | None = None,
    branch_means: dict | None = None,
    save_rng: bool = True,
) -> Path:
    """Write a checkpoint atomically.

    Atomic because a crash (or a full disk) partway through ``torch.save``
    otherwise leaves a truncated file that is indistinguishable from a good one
    until you try to load it -- typically days later, when it is the only copy of
    an expensive run. Write to a temp file in the same directory, then
    ``os.replace``, which is atomic on the same filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = _to_plain(config)
    _assert_weights_only_safe(config, "config")
    if metrics is not None:
        metrics = _to_plain(metrics)
        _assert_weights_only_safe(metrics, "metrics")

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "git": git_revision(),
        "metrics": metrics or {},
        "eer_threshold": float(eer_threshold) if eer_threshold is not None else None,
        "branch_means": branch_means or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if save_rng:
        payload["rng"] = {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on the same filesystem
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of a checkpoint -- the model version surfaced by the API (T54)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path, map_location: str = "cpu") -> dict:
    """Load the payload dict. Always ``weights_only=True`` -- pickle is RCE.

    A checkpoint is an untrusted input the moment it comes from anywhere but your
    own disk (a teammate, a bucket, the Hub). ``weights_only=False`` on a hostile
    file executes arbitrary code at load.
    """
    return torch.load(path, map_location=map_location, weights_only=True)


def load_for_inference(
    path: str | Path,
    map_location: str = "cpu",
    strict: bool = True,
) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from a checkpoint → ``(model.eval(), meta)``.

    **The only checkpoint function the backend calls** (T54). It imports nothing
    but ``ml.models``, so the API never depends on training code.

    The architecture is reconstructed from primitives recorded at save time --
    never guessed, and never hardcoded at the call site. A backend that assumed
    ``fusion="concat"`` would silently mis-load an attention checkpoint's weights.
    """
    from ml.models.classifier import ImageClassifier, VideoClassifier

    ckpt = load_raw(path, map_location=map_location)
    config = ckpt.get("config", {})

    arch = config.get("arch", "image")
    classes = {"image": ImageClassifier, "video": VideoClassifier}
    if arch not in classes:
        raise ValueError(
            f"checkpoint {path} has arch={arch!r}; expected one of {sorted(classes)}"
        )

    model = classes[arch](
        num_classes=config.get("num_classes", 2),
        dropout=config.get("dropout", 0.4),
        fusion=config.get("fusion", "concat"),
        # Never download ImageNet weights here: they are about to be overwritten
        # by the state_dict, so fetching them would be a slow no-op that also
        # makes model loading require network access.
        pretrained=False,
    )
    model.load_state_dict(ckpt["model"], strict=strict)
    model.eval()

    meta = {
        "format_version": ckpt.get("format_version", 0),
        "arch": arch,
        "fusion": config.get("fusion", "concat"),
        "epoch": ckpt.get("epoch"),
        "git": ckpt.get("git", {}),
        "metrics": ckpt.get("metrics", {}),
        "eer_threshold": ckpt.get("eer_threshold"),
        "branch_means": ckpt.get("branch_means", {}),
        "class_names": config.get("class_names", ["real", "fake"]),
        "image_size": config.get("image_size", 224),
        "norm_mean": config.get("norm_mean", [0.485, 0.456, 0.406]),
        "norm_std": config.get("norm_std", [0.229, 0.224, 0.225]),
        # False unless a calibration step (T78) explicitly set it. The API must
        # not present an uncalibrated softmax as a probability (T58), so the
        # honest default is "not calibrated".
        "calibrated": bool(config.get("calibrated", False)),
    }
    return model, meta


def load_for_training(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Restore a run in place → the payload dict (for epoch/step/metrics)."""
    ckpt = load_raw(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)

    for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
        if obj is not None and ckpt.get(key) is not None:
            obj.load_state_dict(ckpt[key])

    if restore_rng and "rng" in ckpt:
        torch.set_rng_state(ckpt["rng"]["torch"].cpu().to(torch.uint8))
        if torch.cuda.is_available() and ckpt["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in ckpt["rng"]["cuda"]]
            )
    return ckpt


def transfer_image_to_video(
    video_model: torch.nn.Module, image_state_dict: dict
) -> tuple[list, list]:
    """Load stage-1 image weights into a stage-2 video model → (missing, unexpected).

    ``VideoClassifier`` subclasses ``DeepfakeClassifier``, so ``spatial.*``,
    ``frequency.*``, ``fusion.*`` and ``classifier.*`` share module paths exactly
    -- only ``temporal.*`` is new. **Raises** unless the missing keys are exactly
    the temporal branch (T33).

    Without that assertion, a renamed module would make ``strict=False`` silently
    match nothing, and stage 2 would train from scratch while reporting that it
    transferred. That failure is invisible: the run completes, the loss goes down,
    and the number is just quietly worse than it should be.
    """
    missing, unexpected = video_model.load_state_dict(image_state_dict, strict=False)
    unexpected = list(unexpected)
    bad_missing = [k for k in missing if not k.startswith("temporal.")]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"image->video transfer did not line up.\n"
            f"  unexpected keys (in checkpoint, not in model): {unexpected[:5]}\n"
            f"  missing keys outside temporal.*: {bad_missing[:5]}\n"
            f"  Expected ONLY temporal.* to be missing. If module paths were "
            f"renamed, stage 2 would silently train from scratch."
        )
    return list(missing), unexpected
