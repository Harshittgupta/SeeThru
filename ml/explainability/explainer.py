"""The explainability facade (BUILD_PLAN T46-T52).

One entry point for the backend (T58)::

    explainer = Explainer(model, meta)
    explanation, artifacts = explainer.explain_image(x)

It owns the state that is easy to get wrong from outside:

* **eval mode**, restored afterwards -- dropout would make the CAM random, and
  permanently eval-ing a caller's training model is rude.
* **gradients on**, despite ``predict()`` being ``@torch.no_grad()`` (T48). Note
  ``torch.enable_grad()`` nests: it re-enables inside a ``no_grad`` caller.
* **degenerate detection** -- a component that failed is reported as failed, never
  rendered. A confident-looking fake heatmap is worse than no heatmap (T53).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ml.explainability import attribution as attribution_mod
from ml.explainability import frequency_viz, temporal_viz
from ml.explainability.contracts import (
    Explanation,
    ExplanationArtifacts,
    decide_verdict,
)
from ml.explainability.gradcam import (
    GradCAM,
    eval_mode,
    is_degenerate,
    spatial_target_layer,
)
from ml.explainability.render import (
    DEFAULT_HEATMAP_ALPHA,
    array_to_png,
    denormalize,
    figure_to_png,
    overlay_heatmap,
    radial_profile_figure,
    spectrum_figure,
    timeline_figure,
)

logger = logging.getLogger(__name__)


class Explainer:
    """Produces an :class:`Explanation` + :class:`ExplanationArtifacts`."""

    def __init__(self, model, meta: dict | None = None, alpha: float = DEFAULT_HEATMAP_ALPHA) -> None:
        self.model = model
        self.meta = meta or {}
        self.alpha = alpha
        # From the checkpoint (T24). Without them ablation falls back to zeros,
        # which is off-manifold and says so loudly (T51).
        self.branch_means = self.meta.get("branch_means") or None
        self.references = self.meta.get("radial_references") or None
        self.calibrated = bool(self.meta.get("calibrated", False))

    # ------------------------------------------------------------------ #
    def explain_image(self, x: torch.Tensor) -> tuple[Explanation, ExplanationArtifacts]:
        """Explain one image → ``(Explanation, artifacts)``."""
        if x.dim() == 3:
            x = x.unsqueeze(0)

        artifacts = ExplanationArtifacts()
        degenerate: dict[str, bool] = {}
        warnings: list[str] = []

        with eval_mode(self.model):
            # Branch features first; the backbone runs ONCE for everything below.
            with torch.no_grad():
                logits, aux = self.model.forward_explain(x)
            p_fake = float(torch.softmax(logits.float(), dim=1)[0, 1])

            # GradCAM needs gradients -- a separate pass, deliberately (T48).
            cam = self._cam(x, degenerate, warnings)

            branch_attr = attribution_mod.branch_attribution(
                self.model, aux, branch_means=self.branch_means
            )
            evidence, spectrum = frequency_viz.frequency_evidence(x, self.references)

        original = denormalize(x)
        if cam is not None:
            # cam is (B, 7, 7); take the single image's map. Passing the batched
            # array straight to overlay_heatmap makes cv2.resize read the leading
            # dim as CHANNELS -> (224, 224, B) -> a broadcast error, or worse, a
            # silently wrong blend for B > 1.
            artifacts.add(
                "heatmap.png",
                array_to_png(overlay_heatmap(original, cam[0], self.alpha)),
            )
            artifacts.add("original.png", array_to_png(original))
        artifacts.add("spectrum.png", figure_to_png(spectrum_figure(spectrum)))
        artifacts.add(
            "radial.png",
            figure_to_png(
                radial_profile_figure(
                    np.asarray(evidence.radial_profile),
                    np.asarray(evidence.reference_real) if evidence.reference_real else None,
                    np.asarray(evidence.reference_fake) if evidence.reference_fake else None,
                )
            ),
        )

        warnings.extend(attribution_mod.describe(branch_attr))
        warnings.extend(frequency_viz.describe(evidence))
        if not self.calibrated:
            warnings.append(
                "This score is uncalibrated -- it is not a probability. Do not "
                "present it as a confidence percentage."
            )

        verdict = decide_verdict(p_fake)
        return (
            Explanation(
                label="fake" if p_fake >= 0.5 else "real",
                verdict=verdict,
                p_fake=p_fake,
                calibrated=self.calibrated,
                attribution=branch_attr,
                frequency=evidence,
                degenerate=degenerate,
                warnings=warnings,
                artifact_names=artifacts.names(),
            ),
            artifacts,
        )

    # ------------------------------------------------------------------ #
    def explain_clip(
        self,
        x: torch.Tensor,
        fps: float = 0.0,
        source_indices: list[int] | None = None,
        interpolated: list[bool] | None = None,
    ) -> tuple[Explanation, ExplanationArtifacts]:
        """Explain a ``(1, T, 3, H, W)`` clip, with the manipulation timeline."""
        if x.dim() == 4:
            x = x.unsqueeze(0)

        artifacts = ExplanationArtifacts()
        degenerate: dict[str, bool] = {}
        warnings: list[str] = []

        with eval_mode(self.model):
            with torch.no_grad():
                logits, aux = self.model.forward_explain(x)
            p_fake = float(torch.softmax(logits.float(), dim=1)[0, 1])

            # One backward gives all T frame CAMs: the backbone flattens (B,T)
            # to (B*T), so the activation batch already IS the frames (T47).
            cams = self._cam(x, degenerate, warnings, expect=x.shape[1])

            branch_attr = attribution_mod.branch_attribution(
                self.model, aux, branch_means=self.branch_means
            )
            frame_p = temporal_viz.per_frame_scores(self.model, aux)

        attention = aux.get("temporal_attn")
        attention_np = attention[0].cpu().numpy() if attention is not None else None

        frames = temporal_viz.build_timeline(
            frame_p,
            attention=attention_np,
            source_indices=source_indices,
            fps=fps,
            interpolated=interpolated,
        )
        spans = temporal_viz.merge_spans(frames)

        artifacts.add(
            "timeline.png",
            figure_to_png(
                timeline_figure(
                    [f.t_seconds if f.t_seconds is not None else float(f.index) for f in frames],
                    [f.p_fake for f in frames],
                    [f.suspicious for f in frames],
                    [f.interpolated for f in frames],
                )
            ),
        )
        if cams is not None:
            for i in range(min(len(cams), x.shape[1])):
                frame_rgb = denormalize(x[0, i])
                artifacts.add(
                    f"cam_f{i:02d}.png",
                    array_to_png(overlay_heatmap(frame_rgb, cams[i], self.alpha)),
                )

        warnings.extend(attribution_mod.describe(branch_attr))
        warnings.extend(temporal_viz.describe(frames, spans))
        if not self.calibrated:
            warnings.append(
                "This score is uncalibrated -- it is not a probability. Do not "
                "present it as a confidence percentage."
            )

        return (
            Explanation(
                label="fake" if p_fake >= 0.5 else "real",
                verdict=decide_verdict(p_fake),
                p_fake=p_fake,
                calibrated=self.calibrated,
                attribution=branch_attr,
                timeline=frames,
                spans=spans,
                degenerate=degenerate,
                warnings=warnings,
                artifact_names=artifacts.names(),
            ),
            artifacts,
        )

    # ------------------------------------------------------------------ #
    def _cam(
        self,
        x: torch.Tensor,
        degenerate: dict,
        warnings: list[str],
        expect: int = 1,
    ) -> np.ndarray | None:
        """GradCAM, or None. Never raises into the caller's request path.

        A failed heatmap must degrade to "no heatmap", not to a 500 and not to a
        map made of noise: the verdict and the attribution are still perfectly
        good without it.
        """
        try:
            with GradCAM(self.model, spatial_target_layer(self.model)) as cam:
                maps = cam(x, target_class=1)
        except (RuntimeError, AttributeError) as exc:
            logger.warning("GradCAM failed: %s", exc)
            degenerate["heatmap"] = True
            warnings.append("A heatmap could not be produced for this input.")
            return None

        dead = [i for i in range(len(maps)) if is_degenerate(maps[i])]
        if len(dead) == len(maps):
            # Min-max normalising a flat CAM amplifies float noise into something
            # that looks exactly like a real explanation. Report, do not render.
            logger.warning("All %d CAMs are degenerate; not rendering a heatmap", len(maps))
            degenerate["heatmap"] = True
            warnings.append(
                "The heatmap carried no signal for this input and has been "
                "omitted rather than shown as a map of noise."
            )
            return None

        degenerate["heatmap"] = False
        if dead:
            warnings.append(f"{len(dead)} frame heatmap(s) carried no signal.")
        return maps
