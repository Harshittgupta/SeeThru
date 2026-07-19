"""Model lifecycle (BUILD_PLAN T54).

Loads weights ONCE at startup and holds them for the process. The traps this
class exists to avoid, each of which was called out in the audit:

* **Never call ``classifier.predict()`` from the server.** It does
  ``self.eval()``/``self.train()`` -- it mutates *shared* module state and does
  not restore on exception. Under a threaded server two requests race on the
  model's mode. We call ``.eval()`` once here and only ever ``forward()`` after.
* **Never serve an untrained head.** ``pretrained=True`` only fills the spatial
  backbone; fusion/classifier/temporal are random. Serving that is confident
  noise. ``/ready`` stays 503 until real weights load, unless
  ``allow_untrained`` is explicitly set (local dev).
* **One GPU lock.** A single CUDA device cannot run two inferences at once
  without OOM risk, and RetinaFace (TensorFlow) shares that device. All GPU work
  -- torch AND face detection -- goes through one async lock.
* **Warm up RetinaFace too.** It lazily builds and globally caches its TF model
  on the first ``detect_faces`` call: a multi-second stall, and not thread-safe.
  Do it once at startup, not inside the first user's request.

The registry deliberately imports from ``ml.checkpoint`` only -- ``load_for_inference``
rebuilds the architecture from the checkpoint, so the backend needs no training
code (T24).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

import torch

logger = logging.getLogger("seethru.registry")


class ModelRegistry:
    """Holds the loaded models + a shared GPU lock. Constructed in lifespan."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.image_model = None
        self.image_meta: dict = {}
        self.video_model = None
        self.video_meta: dict = {}
        self.face_detector = None
        self._ready = False
        # The single serialization point for ALL GPU work (torch + TF).
        self.gpu_lock = asyncio.Lock()

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load weights + warm up. Called once, in lifespan startup.

        Synchronous on purpose: nothing should serve until this finishes, and
        ``/ready`` reports 503 in the meantime.
        """
        from ml import checkpoint

        settings = self.settings
        loaded_any = False

        if settings.image_weights:
            self.image_model, self.image_meta = checkpoint.load_for_inference(
                settings.image_weights, map_location=str(self.device)
            )
            self.image_model = self.image_model.to(self.device).eval()
            self.image_version = self._version(settings.image_weights, self.image_meta)
            logger.info("loaded image model %s", self.image_version)
            loaded_any = True

        if settings.video_weights:
            self.video_model, self.video_meta = checkpoint.load_for_inference(
                settings.video_weights, map_location=str(self.device)
            )
            self.video_model = self.video_model.to(self.device).eval()
            self.video_version = self._version(settings.video_weights, self.video_meta)
            logger.info("loaded video model %s", self.video_version)
            loaded_any = True

        if not loaded_any:
            if not settings.allow_untrained:
                logger.error(
                    "No weights configured (SEETHRU_IMAGE_WEIGHTS / "
                    "SEETHRU_VIDEO_WEIGHTS). Refusing to become ready: an "
                    "untrained head produces confident noise. Set "
                    "SEETHRU_ALLOW_UNTRAINED=true for local dev only."
                )
                return
            logger.warning(
                "allow_untrained=true: serving a randomly-initialised model. "
                "Every verdict is meaningless. Local development only."
            )
            self._load_untrained()

        self._warmup()
        self._ready = True

    def _load_untrained(self) -> None:
        """A random image model, for wiring up the frontend without weights."""
        from ml.models.classifier import ImageClassifier

        self.image_model = ImageClassifier(pretrained=False).to(self.device).eval()
        self.image_meta = {"arch": "image", "fusion": "concat", "calibrated": False}
        self.image_version = "untrained"

    def _warmup(self) -> None:
        """One dummy forward per model + one RetinaFace call.

        The RetinaFace warmup is the point: it builds and caches its TF model on
        first use, so without this the first real request eats a multi-second
        stall and hits a not-thread-safe init.
        """
        with torch.inference_mode():
            if self.image_model is not None:
                self.image_model(torch.randn(1, 3, 224, 224, device=self.device))
            if self.video_model is not None:
                self.video_model(torch.randn(1, 16, 3, 224, 224, device=self.device))
        logger.info("model warmup done")

        try:
            from ml.preprocessing.face_detector import FaceDetector

            self.face_detector = FaceDetector()
            import numpy as np

            # Force the TF model to build now, on a throwaway frame.
            self.face_detector.detect_and_align(
                np.zeros((256, 256, 3), dtype=np.uint8)
            )
            logger.info("RetinaFace warmup done")
        except Exception as exc:  # noqa: BLE001 - detector is optional at boot
            logger.warning("face detector warmup failed (will retry per request): %s", exc)

    @staticmethod
    def _version(path: str, meta: dict) -> str:
        """Content hash + arch, surfaced on every prediction (T58)."""
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            while block := fh.read(1 << 20):
                digest.update(block)
        return f"{meta.get('arch', '?')}-{digest.hexdigest()[:12]}"

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        return self._ready

    def require_image(self):
        if self.image_model is None:
            from backend.core.errors import ModelNotReady

            raise ModelNotReady("The image model is not loaded.")
        return self.image_model

    def require_video(self):
        if self.video_model is None:
            from backend.core.errors import ModelNotReady

            raise ModelNotReady(
                "The video model is not loaded. Video analysis is unavailable."
            )
        return self.video_model

    def model_info(self) -> dict:
        """For GET /v1/model/info and every prediction's `model` block (T58)."""
        return {
            "device": str(self.device),
            "image": {
                "loaded": self.image_model is not None,
                "version": getattr(self, "image_version", None),
                "calibrated": bool(self.image_meta.get("calibrated", False)),
                "class_names": self.image_meta.get("class_names", ["real", "fake"]),
            },
            "video": {
                "loaded": self.video_model is not None,
                "version": getattr(self, "video_version", None),
                "calibrated": bool(self.video_meta.get("calibrated", False)),
            },
            # In-band, non-strippable. FF++-trained detectors generalise poorly,
            # and the caller must be told in the response, not just the docs (T58).
            "disclaimer": {
                "not_forensic_evidence": True,
                "known_limitations": [
                    "degrades on unseen manipulation methods, compression, and resolution",
                    "single-subject videos only",
                    "not validated for legal, journalistic, or evidentiary use",
                ],
            },
        }
