"""Backend test fixtures (BUILD_PLAN T59).

Two traps this file exists to avoid:

* **Use `with TestClient(app) as c:`.** A bare `TestClient(app)` does NOT run the
  lifespan, so the registry is never created and every request 503s. Every client
  fixture here is a context manager.
* **A tiny fake model + fake detector.** Tests must never import TensorFlow, load
  real weights, or need a GPU. The registry is monkeypatched to hold an nn.Module
  that returns fixed logits and a detector that returns a synthetic crop.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch


@pytest.fixture
def fake_image_bytes() -> bytes:
    from PIL import Image

    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


class _FixedModel(torch.nn.Module):
    """An ImageClassifier-shaped stand-in with real explainability plumbing.

    Uses the actual ImageClassifier so forward_explain / fuse_and_classify /
    spatial.features all exist, but warms its BatchNorm stats so the CAM is not
    degenerate (an untrained eval() backbone outputs ~zero -- Milestone 2).
    """

    def __new__(cls):
        from ml.models.classifier import ImageClassifier

        torch.manual_seed(0)
        model = ImageClassifier(pretrained=False)
        model.train()
        with torch.no_grad():
            for _ in range(8):
                model(torch.randn(4, 3, 224, 224))
        model.eval()
        return model


class _FakeDetector:
    """Always returns one synthetic aligned crop + a bbox."""

    def __init__(self, faces: int = 1) -> None:
        self.faces = faces
        self.last_boxes = [[10, 10, 200, 200]][:faces]

    def detect_and_align(self, image):
        crop = np.full((224, 224, 3), 128, dtype=np.uint8)
        self.last_boxes = [[10, 10, 200, 200] for _ in range(self.faces)]
        return [crop for _ in range(self.faces)]


@pytest.fixture
def app_settings(tmp_path, monkeypatch):
    """Settings pointed at tmp dirs, with untrained serving allowed."""
    monkeypatch.setenv("SEETHRU_ALLOW_UNTRAINED", "true")
    monkeypatch.setenv("SEETHRU_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SEETHRU_UPLOAD_TMP_DIR", str(tmp_path / "tmp"))
    from backend.core.config import get_settings

    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def client(app_settings, monkeypatch):
    """A TestClient whose registry is faked. Runs lifespan (context manager)."""
    from fastapi.testclient import TestClient

    from backend.services.registry import ModelRegistry

    def fake_load(self):
        self.image_model = _FixedModel()
        self.image_meta = {"arch": "image", "fusion": "concat", "calibrated": False}
        self.image_version = "image-testfake000"
        self.video_model = None
        self.face_detector = _FakeDetector()
        self._ready = True

    monkeypatch.setattr(ModelRegistry, "load", fake_load)

    from backend.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def client_no_face(app_settings, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.services.registry import ModelRegistry

    class _NoFace(_FakeDetector):
        def detect_and_align(self, image):
            return []

    def fake_load(self):
        self.image_model = _FixedModel()
        self.image_meta = {"arch": "image", "fusion": "concat", "calibrated": False}
        self.image_version = "image-testfake000"
        self.face_detector = _NoFace()
        self._ready = True

    monkeypatch.setattr(ModelRegistry, "load", fake_load)
    from backend.main import create_app

    with TestClient(create_app()) as c:
        yield c
