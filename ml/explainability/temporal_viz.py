"""Temporal explainability: the manipulation timeline (BUILD_PLAN T50).

**The spec's "Attention Threshold: 0.6" cannot fire, and this is measured.**

``TemporalAttention`` returns a **softmax over time**, so a clip's 16 weights sum
to 1 and a uniform weight is 1/16 = 0.0625. Over 200 random clips the *maximum*
weight seen in any clip was **0.0662**; a raw ``>= 0.6`` test flagged **0/200**.
It would flag zero frames forever, silently, and look like "no manipulation
detected" rather than "this threshold is unreachable".

So the threshold is applied to **normalized** attention, ``w / w.max()``: the
most-attended frame is always 1.0 and 0.6 means "at least 60% as attended as the
peak". The human-readable form is ``w * T`` -- *"3.2x more attention than
average"* -- which is what the UI should show.

**Per-frame scores did not exist.** ``VideoClassifier`` emits one clip-level
logit. But ``forward_explain`` already returns ``spatial_seq (B,T,1536)`` and
``frequency_seq (B,T,128)``, so re-running fusion+head per frame (with
``temporal=None``) gives a per-frame P(fake) for the cost of 16 tiny MLP passes.
The backbone does not run again.

Be honest about what that number is: the model was trained on **clip-level**
labels, so a per-frame score is an uncalibrated proxy for "how fake does this
frame look in isolation", not a per-frame ground truth.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ml.explainability.contracts import FrameScore, TimelineSpan

logger = logging.getLogger(__name__)

#: Applied to NORMALIZED attention (w / w.max()), never to the raw softmax.
ATTENTION_THRESHOLD = 0.6
#: P(fake) above which a frame is suspicious. A frame must clear BOTH this and
#: the attention threshold.
SUSPICIOUS_P_FAKE = 0.6
#: A span needs at least this many CONSECUTIVE suspicious samples. With 16
#: samples over a whole video, one isolated point is not a region.
MIN_SPAN_FRAMES = 2


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """``w / w.max()`` → [0, 1], peak = 1.0.

    This is what makes a 0.6 threshold meaningful. See the module docstring for
    why the raw softmax cannot be thresholded at 0.6.
    """
    peak = float(attention.max())
    if peak <= 0:
        return np.zeros_like(attention)
    return attention / peak


def attention_relative_to_uniform(attention: np.ndarray) -> np.ndarray:
    """``w * T`` -- "Nx more attention than average". The number to show a human.

    1.0 means "exactly average". It is interpretable without knowing what a
    softmax is, which the raw 0.0625 is not.
    """
    return attention * len(attention)


@torch.no_grad()
def per_frame_scores(model, aux: dict, target_class: int = 1) -> np.ndarray:
    """P(fake) for each frame → ``(T,)``.

    Reuses the already-computed per-frame branch features and runs only
    fusion+head, with ``temporal=None`` (a single frame has no temporal context).
    ~16 tiny MLP passes; the backbone does not re-run.
    """
    spatial_seq = aux.get("spatial_seq")
    frequency_seq = aux.get("frequency_seq")
    if spatial_seq is None or frequency_seq is None:
        raise ValueError(
            "per_frame_scores needs spatial_seq/frequency_seq from "
            "VideoClassifier.forward_explain -- an image model has no timeline."
        )

    # (1, T, C) -> (T, C): score every frame as its own batch item.
    spatial = spatial_seq[0]
    frequency = frequency_seq[0]
    logits = model.fuse_and_classify(spatial, frequency, None)
    return torch.softmax(logits.float(), dim=1)[:, target_class].cpu().numpy()


def build_timeline(
    p_fake: np.ndarray,
    attention: np.ndarray | None = None,
    source_indices: list[int] | None = None,
    fps: float = 0.0,
    interpolated: list[bool] | None = None,
) -> list[FrameScore]:
    """Assemble the per-frame timeline.

    Args:
        p_fake: ``(T,)`` per-frame fake probability.
        attention: ``(T,)`` raw softmax-over-time weights, if available.
        source_indices: true frame index of each sample, for real timestamps (T40).
        fps: from the video. 0 → timestamps unavailable, and reported as None
            rather than faked from the frame index.
        interpolated: per-frame, whether the face was copied from a neighbour.
    """
    n = len(p_fake)
    source_indices = source_indices or list(range(n))
    interpolated = interpolated or [False] * n

    norm = normalize_attention(attention) if attention is not None else None

    frames: list[FrameScore] = []
    for i in range(n):
        attention_norm = float(norm[i]) if norm is not None else None
        # BOTH conditions. A high-attention frame the model finds unremarkable is
        # not suspicious, and a high-p_fake frame the temporal branch ignored is
        # not what the timeline is claiming to show.
        suspicious = bool(
            p_fake[i] >= SUSPICIOUS_P_FAKE
            and (attention_norm is None or attention_norm >= ATTENTION_THRESHOLD)
            # An interpolated frame is a copy of a neighbour, not an
            # observation. It must never be flagged as evidence.
            and not interpolated[i]
        )
        frames.append(
            FrameScore(
                index=i,
                source_index=int(source_indices[i]),
                p_fake=float(p_fake[i]),
                t_seconds=(source_indices[i] / fps) if fps > 0 else None,
                attention=float(attention[i]) if attention is not None else None,
                attention_norm=attention_norm,
                suspicious=suspicious,
                interpolated=bool(interpolated[i]),
            )
        )
    return frames


def merge_spans(frames: list[FrameScore], min_frames: int = MIN_SPAN_FRAMES) -> list[TimelineSpan]:
    """Merge consecutive suspicious frames into spans.

    Requires >= ``min_frames`` **consecutive** samples. With 16 samples spread
    across a whole video, neighbouring samples can be ~19 s apart on a 5-minute
    clip -- so a span is a *sampled region*, not a continuous detection, and one
    isolated point is not a region at all (T62).
    """
    spans: list[TimelineSpan] = []
    run: list[FrameScore] = []

    def flush() -> None:
        if len(run) >= min_frames:
            times = [f.t_seconds for f in run if f.t_seconds is not None]
            spans.append(
                TimelineSpan(
                    start_s=min(times) if times else float(run[0].index),
                    end_s=max(times) if times else float(run[-1].index),
                    mean_p_fake=float(np.mean([f.p_fake for f in run])),
                    n_frames=len(run),
                )
            )
        run.clear()

    for frame in frames:
        if frame.suspicious:
            run.append(frame)
        else:
            flush()
    flush()
    return spans


def describe(frames: list[FrameScore], spans: list[TimelineSpan]) -> list[str]:
    """Plain sentences for the timeline."""
    if not frames:
        return []

    lines: list[str] = []
    n_interp = sum(f.interpolated for f in frames)
    have_time = any(f.t_seconds is not None for f in frames)

    if not spans:
        lines.append("No sustained suspicious region was found in the sampled frames.")
    else:
        for span in spans:
            if have_time:
                lines.append(
                    f"{span.start_s:.1f}s-{span.end_s:.1f}s: suspicious artifacts "
                    f"across {span.n_frames} sampled frames "
                    f"(mean score {span.mean_p_fake:.2f})."
                )
            else:
                lines.append(
                    f"{span.n_frames} consecutive sampled frames look suspicious "
                    f"(mean score {span.mean_p_fake:.2f})."
                )

    # Always state the sampling caveat. Without it "0-2s real, 2-5s suspicious"
    # implies continuous analysis, which is simply not what happened.
    lines.append(
        f"Based on {len(frames)} frames sampled across the video, not a "
        f"continuous analysis."
    )
    if n_interp:
        lines.append(
            f"{n_interp} of those frames had no detectable face and were filled "
            f"from a neighbour -- they are not independent evidence."
        )
    if not have_time:
        lines.append("Timestamps unavailable (the video reported no frame rate).")
    return lines
