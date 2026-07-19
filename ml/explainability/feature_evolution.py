"""Feature-evolution visualization (BUILD_PLAN T52).

The spec asks for: Original → Edge → Texture → Frequency → Attention → Prediction.

**Honest assessment, because this one invites dishonesty.** The obvious way to
build it is `cv2.Sobel` and `cv2.Canny` panels captioned "what the model sees".
That would be **theater**: the model never computes Sobel or Canny. It would be a
picture of an edge detector, presented as a window into a network that contains
no edge detector. For a project whose entire thesis is honest explanation, that
is the worst possible thing to ship -- it is not a harmless simplification, it is
a fabricated explanation, and it is *more* persuasive than the real one.

The ladder is salvageable, though, because a CNN genuinely does build features in
roughly that order. So each rung comes from a **real intermediate activation**:

    Original    the input, denormalized
    Edge        spatial.features[0]  -- the stem Conv2dNormActivation at 112x112.
                Genuinely an edge/colour-blob detector; this is what the first
                layer of a convnet is.
    Texture     spatial.features[2]  -- mid-stage, 28x28. Composite local
                patterns.
    Frequency   the model's OWN log-FFT spectrum (the frequency branch's input),
                not a decorative FFT computed differently here.
    Attention   the GradCAM over the last conv stage -- the real one from T47.
    Prediction  the verdict.

Same visual as the spec asked for. Every panel is something the network actually
computed.

Rendered as **one horizontal strip** (a single Agg figure), not six requests:
six round-trips for one explanation is a lot of latency for a picture.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: (attribute path, human label). Chosen because they are what the stages ARE,
#: not because they are convenient: features[0] is the stem conv (edges),
#: features[2] is an early MBConv stack (texture).
LADDER = (
    (0, "Edge (stem conv)"),
    (2, "Texture (mid stage)"),
)


class _ActivationCapture:
    """Capture named intermediate activations in one forward pass."""

    def __init__(self, model: nn.Module, stages: tuple = LADDER) -> None:
        self.backbone = model.spatial.features
        self.stages = stages
        self.acts: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> _ActivationCapture:
        for idx, label in self.stages:
            self._handles.append(
                self.backbone[idx].register_forward_hook(self._make_hook(label))
            )
        return self

    def _make_hook(self, label: str):
        def hook(_m, _i, output):
            self.acts[label] = output.detach()
        return hook

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _activation_image(act: torch.Tensor) -> np.ndarray:
    """A (C, H, W) activation → a viewable (H, W) float map in [0, 1].

    Mean over channels: a single channel is an arbitrary slice of a
    high-dimensional representation, and picking one would imply a significance
    it does not have. The mean is what "how strongly did this stage respond
    here" actually looks like.
    """
    if act.dim() == 4:
        act = act[0]
    m = act.float().mean(dim=0).cpu().numpy()
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-8:
        return np.zeros_like(m)  # dead stage: show nothing, invent nothing
    return (m - lo) / (hi - lo)


@torch.no_grad()
def capture_ladder(model, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Run one forward pass and return the real intermediate activations.

    NOTE: needs the model in a mode where the backbone carries signal. An
    untrained EfficientNet in eval() outputs ~zero (std 7.4e-15 -- Milestone 2),
    so every panel would be a flat grey rectangle. That is honest, but useless;
    with real weights it is fine.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    with _ActivationCapture(model) as capture:
        model.spatial(x)
    return {label: _activation_image(act) for label, act in capture.acts.items()}


def evolution_strip(
    model,
    x: torch.Tensor,
    cam: np.ndarray | None = None,
    verdict: str = "",
    p_fake: float = 0.0,
):
    """The 6-panel strip → a matplotlib Figure (caller renders it to PNG).

    Args:
        model: an ImageClassifier/VideoClassifier.
        x: one normalized ``(3, H, W)`` model input.
        cam: the GradCAM from T47. Passed in rather than recomputed -- it needs
            gradients, and this function is under no_grad.
        verdict / p_fake: for the final panel.
    """
    import matplotlib.pyplot as plt

    from ml.explainability.frequency_viz import log_spectrum
    from ml.explainability.render import denormalize

    original = denormalize(x)
    ladder = capture_ladder(model, x)
    spectrum = log_spectrum(x)

    panels: list[tuple[str, np.ndarray, str]] = [("Original", original, "")]
    for _idx, label in LADDER:
        if label in ladder:
            panels.append((label, ladder[label], "magma"))
    panels.append(("Frequency (model's FFT)", spectrum, "viridis"))
    if cam is not None:
        panels.append(("Attention (GradCAM)", cam, "jet"))

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(2.1 * (len(panels) + 1), 2.6))
    for ax, (title, data, cmap) in zip(axes[:-1], panels, strict=False):
        ax.imshow(data) if not cmap else ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Final panel: the verdict, as text.
    final = axes[-1]
    final.axis("off")
    colour = {"fake": "#e76f51", "real": "#2a9d8f"}.get(verdict, "#6c757d")
    final.text(0.5, 0.58, verdict.upper() or "?", ha="center", va="center",
               fontsize=15, color=colour, weight="bold")
    final.text(0.5, 0.36, f"score {p_fake:.2f}", ha="center", va="center", fontsize=9)
    final.set_title("Prediction", fontsize=8)

    fig.suptitle(
        "How the model processed this image (real intermediate activations)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig
