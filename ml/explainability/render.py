"""Rendering helpers for explainability (BUILD_PLAN T46).

**The `matplotlib.use("Agg")` on the first line is load-bearing.** It must run
before *anything* imports `pyplot`. Without it matplotlib picks a GUI backend
(TkAgg on Windows), and inside a FastAPI worker that either crashes outright or
opens a window nobody will ever close. Agg is a headless raster backend: no
display, no event loop.

Every figure is closed in a `finally`. pyplot keeps a global registry of open
figures, so a leaked figure is never garbage collected -- a server rendering one
explanation per request would climb until it died, over hours, with no obvious
cause.

Returns PNG **bytes**, never file paths: the backend decides where artifacts live
(T58), and a library that writes files is a library that has assumed it owns a
filesystem.
"""

from __future__ import annotations

import io

import matplotlib

# MUST precede any pyplot import, including transitive ones. Do not move.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Heatmap opacity, per the spec (0.4-0.6). NOTE this is the HEATMAP's alpha; the
# `grad-cam` library's `image_weight` is the inverse (the image's weight), which
# is an easy way to get 0.4 where you meant 0.6.
DEFAULT_HEATMAP_ALPHA = 0.5
MIN_ALPHA, MAX_ALPHA = 0.4, 0.6


def figure_to_png(fig) -> bytes:
    """Render a figure to PNG bytes and close it."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)  # always: pyplot holds a global ref, so a leak is forever


def array_to_png(arr: np.ndarray) -> bytes:
    """Encode an HWC uint8 RGB array as PNG bytes, without matplotlib."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def overlay_heatmap(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = DEFAULT_HEATMAP_ALPHA,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a [0,1] CAM over an RGB uint8 image → HWC uint8.

    The CAM is upsampled with **bilinear** interpolation, deliberately. GradCAM
    on EfficientNet-B3 at 224 input produces a 7x7 map -- a 32x upscale. Nearest
    would render hard 32-pixel blocks, which look like precise localisation and
    are nothing of the sort. Bilinear is honest about the resolution it has.
    """
    if not MIN_ALPHA <= alpha <= MAX_ALPHA:
        raise ValueError(
            f"alpha {alpha} outside the spec's 0.4-0.6 heatmap opacity band"
        )
    # Reject a batched CAM explicitly. cv2.resize reads a 3D array as HWC, so a
    # (1, 7, 7) batch silently becomes a (224, 224, 7) "7-channel image" -- which
    # then either fails to broadcast or, for the wrong shapes, blends garbage
    # without complaint. Caught exactly this way in T53.
    if cam.ndim != 2:
        raise ValueError(
            f"overlay_heatmap expects a single 2-D CAM, got shape {cam.shape}. "
            f"GradCAM returns (B, h, w) -- index the frame you mean, e.g. cam[0]."
        )

    import cv2

    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    heat = (plt.get_cmap(colormap)(cam_resized)[..., :3] * 255).astype(np.uint8)
    blended = (alpha * heat.astype(np.float32) + (1 - alpha) * image_rgb.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def denormalize(
    tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Undo ImageNet normalization → HWC uint8 RGB, for display.

    The model's input is normalized; a human needs the original pixels back to
    see what the heatmap is pointing at.
    """
    import torch

    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float()
        if arr.dim() == 4:
            arr = arr[0]
        arr = arr.permute(1, 2, 0).numpy()
    else:
        arr = np.asarray(tensor)

    arr = arr * np.asarray(std) + np.asarray(mean)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def spectrum_figure(spectrum: np.ndarray, title: str = "Log-magnitude FFT spectrum"):
    """A single-panel spectrum heatmap. Decorative, and labelled as such (T49)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(spectrum, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def radial_profile_figure(
    profile: np.ndarray,
    reference_real: np.ndarray | None = None,
    reference_fake: np.ndarray | None = None,
):
    """Radial power profile against reference bands (T49).

    The reference curves are the point. A lone radial profile is an unlabelled
    squiggle -- a viewer has no way to know whether its high-frequency tail is
    unusual. Plotted against the training-set means for real and fake, the same
    curve becomes readable at a glance.
    """
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.linspace(0, 1, len(profile))

    if reference_real is not None and len(reference_real):
        ax.plot(np.linspace(0, 1, len(reference_real)), reference_real,
                "--", color="#2a9d8f", label="typical real", linewidth=1.2)
    if reference_fake is not None and len(reference_fake):
        ax.plot(np.linspace(0, 1, len(reference_fake)), reference_fake,
                "--", color="#e76f51", label="typical fake", linewidth=1.2)

    ax.plot(x, profile, color="#264653", label="this image", linewidth=2)
    ax.set_xlabel("spatial frequency (0 = DC, 1 = Nyquist)", fontsize=9)
    ax.set_ylabel("log magnitude", fontsize=9)
    ax.set_title("Radial frequency profile", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def timeline_figure(
    t_seconds: list[float],
    p_fake: list[float],
    suspicious: list[bool],
    interpolated: list[bool],
    threshold: float = 0.5,
):
    """The manipulation timeline (T50).

    **Discrete markers on a seconds axis, never a continuous line.** 16 samples
    spread across a whole video means adjacent points can be ~19 s apart on a
    5-minute clip; joining them with a line asserts a continuity that was never
    measured. Interpolated frames are drawn hollow -- they are copies of a
    neighbour, not observations, and a filled marker would claim otherwise.
    """
    fig, ax = plt.subplots(figsize=(7, 2.6))

    ax.axhline(threshold, color="#999", linestyle=":", linewidth=1, zorder=1)

    for t, p, susp, interp in zip(t_seconds, p_fake, suspicious, interpolated, strict=True):
        if interp:
            ax.plot(t, p, "o", mfc="none", mec="#888", ms=7, zorder=3)
        else:
            ax.plot(t, p, "o", color="#e76f51" if susp else "#2a9d8f", ms=7, zorder=3)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time (seconds)", fontsize=9)
    ax.set_ylabel("P(fake)", fontsize=9)
    ax.set_title("Per-frame score (sampled points, not a continuous signal)", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#2a9d8f", label="scored"),
            Line2D([], [], marker="o", ls="", color="#e76f51", label="suspicious"),
            Line2D([], [], marker="o", ls="", mfc="none", mec="#888", label="interpolated (not measured)"),
        ],
        fontsize=7, loc="upper right", framealpha=0.9,
    )
    fig.tight_layout()
    return fig
