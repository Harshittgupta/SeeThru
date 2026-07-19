"""GradCAM for SEETHRU (BUILD_PLAN T47/T48).

**Hand-rolled, ~40 lines. `grad-cam` (jacobgil) was evaluated and rejected.**

Reasons, verified against its installed source rather than assumed:

* ``BaseCAM.get_target_width_height`` reads a 5D input as a *3D conv volume*, so
  a ``(1, 16, 3, 224, 224)`` clip yields a target size of ``(224, 224, 3)`` and
  ``scale_cam_image`` returns garbage. The video path is structurally broken.
* It assumes activation-batch == input-batch. ``VideoClassifier.forward``
  flattens ``(B, T, C, H, W) -> (B*T, C, H, W)`` before the backbone, so
  activations come back with batch ``B*T`` while the input batch is ``B``, and
  ``aggregate_multi_layers`` returns 16 CAMs for 1 input.
* ``BaseCAM.__init__`` silently mutates the caller's model (``self.model =
  model.eval()``).

Maintaining two CAM paths -- library for images, hand-rolled for video -- is
worse than owning 40 lines that work for both. `grad-cam` is dropped from
requirements.

**Target layer: `model.spatial.features[-1]`.** Verified as `features[8]`, the
final ``Conv2dNormActivation(Conv2d, BatchNorm2d, SiLU)``, activation
``(B, 1536, 7, 7)``. Hook the whole block, not ``features[8][0]`` -- the raw conv
output is pre-BN/pre-activation and is not what the network actually propagates.

**7x7 upsampled to 224 is a 32x scale-up.** The map is genuinely coarse. Say so
in the UI; do not sell it as pixel-level localisation.

**Spatial branch only.** No frequency GradCAM: that CNN's H/W axes are FFT
coordinates, not image coordinates, so overlaying its CAM on a face is actively
misleading. The radial profile (T49) says the same thing legibly.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Below this, a CAM is flat and carries no information. Min-max normalising a
# flat map amplifies float noise into something that LOOKS like an explanation,
# which is worse than returning nothing (T53).
DEGENERATE_RANGE = 1e-6


class GradCAM:
    """Grad-weighted class activation mapping over a target conv layer.

    Usage::

        with GradCAM(model, model.spatial.features[-1]) as cam:
            maps = cam(x, target_class=1)   # (B, 7, 7) in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    # ------------------------------------------------------------------ #
    def _forward_hook(self, _module, _inputs, output) -> None:
        self._activations = output
        # Register the backward hook on the TENSOR, not the module. Module
        # backward hooks are unreliable with in-place ops, and EfficientNet's
        # SiLU is inplace.
        if output.requires_grad:
            self._handles.append(output.register_hook(self._save_gradient))

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def __enter__(self) -> GradCAM:
        self._handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ #
    def __call__(
        self, x: torch.Tensor, target_class: int = 1, normalize: bool = True
    ) -> np.ndarray:
        """→ ``(B, H', W')`` CAMs in [0, 1], one per item in the ACTIVATION batch.

        For a video clip the activation batch is ``B*T``, so a single backward
        yields all 16 frame CAMs at once -- no per-frame loop. That is a
        consequence of the flatten in ``VideoClassifier.forward``, and the reason
        the library's batch assumption breaks while this does not.
        """
        self._activations = self._gradients = None

        # See the class docstring for why grad is forced on here (T48).
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            score = logits[:, target_class].sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()

        self._assert_captured()

        activations = self._activations.detach()      # (N, C, h, w)
        gradients = self._gradients.detach()          # (N, C, h, w)

        # Channel weights = spatially-averaged gradients -> "how much does this
        # channel matter for the target class".
        weights = gradients.mean(dim=(2, 3), keepdim=True)      # (N, C, 1, 1)
        cam = (weights * activations).sum(dim=1)               # (N, h, w)
        cam = torch.relu(cam)                                   # only positive evidence

        cam_np = cam.float().cpu().numpy()
        return _normalize_cams(cam_np) if normalize else cam_np

    def _assert_captured(self) -> None:
        """Fail loudly when the hooks caught nothing (T48).

        This is the guard that matters. If the backbone is frozen
        (``requires_grad_(False)``, exactly what the freeze schedule does for the
        first epochs) **and** the input does not require grad, the activation does
        not require grad, the backward hook never fires, and ``self._gradients``
        stays None. Measured: 0 gradients captured, no exception raised.

        Without this assert that becomes an empty or garbage CAM presented as an
        explanation. `__call__` forces ``requires_grad_`` on the input to prevent
        it; this catches any path that still slips through.
        """
        if self._activations is None:
            raise RuntimeError(
                "GradCAM captured no activations. Is the target layer actually "
                "part of the model that ran? (Hooking a module from a different "
                "instance is silent.)"
            )
        if self._gradients is None:
            raise RuntimeError(
                "GradCAM captured no gradients: the target layer's output did not "
                "require grad. This happens when the backbone is frozen "
                "(requires_grad=False) AND the input does not require grad -- "
                "neither of which raises on its own. Ensure the input has "
                "requires_grad_(True) and that you are not inside torch.no_grad()."
            )


def _normalize_cams(cams: np.ndarray) -> np.ndarray:
    """Min-max each CAM to [0, 1], but return zeros for degenerate maps.

    The standard trick is ``cam / (cam.max() + 1e-7)``. On a dead, flat CAM that
    divides noise by noise and produces a vivid, structured-looking map from
    nothing at all -- an explanation of a model that explained nothing. Callers
    check :func:`is_degenerate` and report rather than render (T53).
    """
    out = np.empty_like(cams, dtype=np.float32)
    for i, cam in enumerate(cams):
        lo, hi = float(cam.min()), float(cam.max())
        if hi - lo < DEGENERATE_RANGE:
            out[i] = np.zeros_like(cam)
            continue
        out[i] = (cam - lo) / (hi - lo)
    return out


def is_degenerate(cam: np.ndarray) -> bool:
    """True when a CAM carries no information (flat, or all zero)."""
    return float(cam.max()) - float(cam.min()) < DEGENERATE_RANGE


def spatial_target_layer(model: nn.Module) -> nn.Module:
    """The GradCAM target: the spatial backbone's last conv stage.

    Verified: ``features[8]`` is a ``Conv2dNormActivation(Conv2d, BatchNorm2d,
    SiLU)`` producing ``(B, 1536, 7, 7)``.
    """
    backbone = getattr(model, "spatial", None)
    if backbone is None or not hasattr(backbone, "features"):
        raise AttributeError(
            f"{type(model).__name__} has no .spatial.features -- GradCAM needs the "
            f"EfficientNet backbone to hook."
        )
    return backbone.features[-1]


@contextmanager
def eval_mode(model: nn.Module):
    """Temporarily eval() the model, restoring the previous mode.

    Explanation must not flip a training model into eval permanently, and must
    not leave it in train mode either (dropout would make the CAM random).
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()
