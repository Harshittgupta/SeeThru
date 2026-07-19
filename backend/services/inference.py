"""Inference orchestration (BUILD_PLAN T56/T57).

Wraps the ML pipeline for the API. Never imports FastAPI -- it raises the typed
``ApiError`` subclasses and returns plain dicts, so it is testable without a
server and reusable from the job worker.

The failure taxonomy is the interesting part, because the honest answers are the
uncommon-looking ones:

* **No face -> 422 ``no_face_detected``.** Not a real/fake guess. The model is
  only meaningful on an aligned crop; "I could not find a face" is a different
  statement from "this face is real", and collapsing them fabricates a result.
* **Multiple faces -> analyse all**, return a ``faces[]`` array. Never silently
  pick face 0 and call it "the" answer.
* **GPU OOM -> 503 ``gpu_busy``** + empty_cache, not a 500. It is transient.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from backend.core.errors import (
    GpuBusy,
    NoFaceDetected,
    UnreadableMedia,
)
from ml.explainability import Explainer
from ml.explainability.contracts import decide_verdict

logger = logging.getLogger("seethru.inference")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_model_input(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """A 224x224 BGR crop → a normalized (1,3,224,224) tensor on device."""
    import cv2

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(IMAGENET_MEAN)) / np.asarray(IMAGENET_STD)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(device)


def _summary(verdict: str, p_fake: float, reasoning: list[str], any_fake: bool) -> dict:
    return {
        "verdict": verdict,
        "confidence": abs(p_fake - 0.5) * 2,  # distance from the boundary, 0..1
        "reasoning": reasoning,
        "any_fake": any_fake,
    }


class InferenceService:
    """Runs image/video analysis behind the registry's GPU lock."""

    def __init__(self, registry, settings) -> None:
        self.registry = registry
        self.settings = settings

    # ------------------------------------------------------------------ #
    async def analyze_image(self, image_bgr: np.ndarray, artifact_dir: Path) -> dict:
        """Detect every face, analyse each, write artifacts → the response dict.

        Runs under the single GPU lock: torch AND RetinaFace share the device.
        """
        model = self.registry.require_image()
        detector = self.registry.face_detector

        async with self.registry.gpu_lock:
            faces = self._detect(detector, image_bgr)
            if not faces:
                # 422, not a verdict. See the module docstring.
                raise NoFaceDetected(
                    "No face was found in the image. SEETHRU only analyses faces."
                )

            explainer = Explainer(model, self.registry.image_meta)
            results = []
            any_fake = False
            for i, (crop, bbox) in enumerate(faces[: self.settings.max_faces]):
                try:
                    exp, artifacts = self._explain_image(explainer, crop)
                except torch.cuda.OutOfMemoryError as exc:  # transient
                    torch.cuda.empty_cache()
                    raise GpuBusy("The GPU is momentarily out of memory. Retry.",
                                  {"retry_after": 5}) from exc

                names = self._write_artifacts(artifacts, artifact_dir, prefix=f"face{i}")
                any_fake = any_fake or exp.verdict == "fake"
                results.append(self._face_result(i, bbox, exp, names))

        verdict, p_fake, reasoning = self._aggregate(results)
        warnings = []
        if len(faces) > self.settings.max_faces:
            warnings.append(
                f"{len(faces)} faces found; only the first {self.settings.max_faces} were analysed."
            )
        return {
            "faces": results,
            "summary": _summary(verdict, p_fake, reasoning, any_fake),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ #
    def _detect(self, detector, image_bgr: np.ndarray):
        """→ list of (crop, bbox). bbox is in ORIGINAL image pixels (T62)."""
        if detector is None:
            raise UnreadableMedia("The face detector is unavailable.")
        # detect_and_align returns aligned crops; we also want the box for the UI.
        crops = detector.detect_and_align(image_bgr)
        boxes = getattr(detector, "last_boxes", None) or [[0, 0, 0, 0]] * len(crops)
        return list(zip(crops, boxes, strict=False))

    def _explain_image(self, explainer, crop_bgr: np.ndarray):
        x = _to_model_input(crop_bgr, self.registry.device)
        return explainer.explain_image(x)

    @staticmethod
    def _write_artifacts(artifacts, artifact_dir: Path, prefix: str) -> dict:
        """PNG bytes → files on disk; return name -> relative URL (T58)."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        urls = {}
        for name, png in artifacts.images.items():
            fname = f"{prefix}_{name}"
            (artifact_dir / fname).write_bytes(png)
            urls[name] = fname  # the route builds the full /v1/artifacts URL
        return urls

    def _face_result(self, i: int, bbox, exp, artifact_names: dict) -> dict:
        return {
            "face_id": i,
            "bbox": [int(v) for v in bbox],
            "verdict": exp.verdict,
            "scores": {"real": 1 - exp.p_fake, "fake": exp.p_fake},
            "attribution": [a.to_dict() for a in exp.attribution],
            "artifacts": artifact_names,
            "hf_energy_ratio": exp.frequency.hf_energy_ratio if exp.frequency else None,
        }

    @staticmethod
    def _aggregate(results: list[dict]) -> tuple[str, float, list[str]]:
        """Overall verdict from per-face results.

        If any face is fake, the media is suspicious -- one manipulated face is
        the whole point. Otherwise uncertain beats a confident 'real'.
        """
        if not results:
            return "uncertain", 0.5, ["No faces analysed."]
        fake_probs = [r["scores"]["fake"] for r in results]
        p_fake = max(fake_probs)  # most-suspicious face drives the headline
        verdict = decide_verdict(p_fake)
        n_fake = sum(r["verdict"] == "fake" for r in results)
        reasoning = [
            f"{len(results)} face(s) analysed; {n_fake} flagged as manipulated."
        ]
        return verdict, p_fake, reasoning


def build_media_block(kind: str, sha256: str, extra: dict | None = None) -> dict:
    block = {"kind": kind, "sha256": sha256}
    block.update(extra or {})
    return block
