"""Backend API tests (BUILD_PLAN T59).

Covers the contract and, more importantly, the honest-failure and security paths:
health-without-model, no-face-is-422-not-a-guess, magic-byte rejection, oversize
rejection, path traversal, and the uncalibrated disclaimer.
"""

from __future__ import annotations

import io

import numpy as np


def _jpeg(size=(64, 64)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((*size, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Health / readiness (T54)
# --------------------------------------------------------------------------- #
def test_health_is_green_without_touching_the_model(client):
    """Liveness must not depend on the model, or a slow load reads as a crash."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_reports_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_health_does_not_require_readiness():
    """The distinction, proven: /health is 200 even when the registry is NOT ready.

    Uses a registry whose load() leaves _ready False -- exactly the mid-startup
    state an orchestrator must not kill.
    """
    import os

    from fastapi.testclient import TestClient

    os.environ["SEETHRU_ALLOW_UNTRAINED"] = "true"
    from backend.core.config import get_settings
    from backend.services.registry import ModelRegistry

    get_settings.cache_clear()
    orig = ModelRegistry.load
    ModelRegistry.load = lambda self: setattr(self, "_ready", False)
    try:
        from backend.main import create_app

        with TestClient(create_app()) as c:
            assert c.get("/health").status_code == 200      # liveness OK
            assert c.get("/ready").status_code == 503        # not ready
    finally:
        ModelRegistry.load = orig
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Image prediction (T56)
# --------------------------------------------------------------------------- #
def test_predict_image_returns_the_contract(client):
    r = client.post("/v1/predict/image", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["summary"]["verdict"] in ("real", "fake", "uncertain")
    assert body["faces"], "at least one face result expected"
    face = body["faces"][0]
    assert set(face["scores"]) == {"real", "fake"}
    assert len(face["bbox"]) == 4
    # Attribution is causal deltas, not shares (ADR 0001).
    assert all("delta" in a for a in face["attribution"])


def test_image_artifacts_are_urls_not_base64(client):
    """T58: the JSON stays small; PNGs stream from /v1/artifacts."""
    r = client.post("/v1/predict/image", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    body = r.json()
    assert len(r.content) < 20_000, "explanation JSON should be small"
    for face in body["faces"]:
        for url in face["artifacts"].values():
            assert url.startswith("/v1/artifacts/")


def test_artifact_urls_actually_serve(client):
    r = client.post("/v1/predict/image", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    urls = list(r.json()["faces"][0]["artifacts"].values())
    if urls:  # a degenerate CAM may omit the heatmap (T53), so guard
        got = client.get(urls[0])
        assert got.status_code == 200
        assert got.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert "immutable" in got.headers.get("cache-control", "")


def test_uncalibrated_disclaimer_is_present(client):
    """The model is uncalibrated (T78); the response must say so in-band (T58)."""
    r = client.post("/v1/predict/image", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    body = r.json()
    assert body["model"]["calibrated"] is False
    assert body["disclaimer"]["not_forensic_evidence"] is True
    assert body["disclaimer"]["known_limitations"]


# --------------------------------------------------------------------------- #
# The honest-failure paths (T56)
# --------------------------------------------------------------------------- #
def test_no_face_is_422_not_a_verdict(client_no_face):
    """The single most important failure behaviour.

    No face must be a distinct 422, never a real/fake guess -- the model is only
    meaningful on an aligned crop.
    """
    r = client_no_face.post("/v1/predict/image", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 422
    assert r.json()["error_code"] == "no_face_detected"


def test_error_envelope_has_request_id(client_no_face):
    r = client_no_face.post("/v1/predict/image", files={"file": ("x.jpg", _jpeg(), "image/jpeg")})
    body = r.json()
    assert set(body) >= {"error_code", "message", "request_id"}
    assert body["request_id"] == r.headers["X-Request-ID"]


# --------------------------------------------------------------------------- #
# Upload security (T55)
# --------------------------------------------------------------------------- #
def test_wrong_magic_bytes_are_rejected(client):
    """A .jpg carrying non-image bytes is rejected on content, not name."""
    r = client.post(
        "/v1/predict/image",
        files={"file": ("evil.jpg", b"this is not an image at all", "image/jpeg")},
    )
    assert r.status_code == 415
    assert r.json()["error_code"] == "unsupported_media"


def test_oversize_upload_is_rejected(client, monkeypatch):
    """Streaming size cap: a body over the limit is aborted mid-read (T55)."""
    from backend.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SEETHRU_MAX_IMAGE_BYTES", "1024")  # 1 KB
    get_settings.cache_clear()

    big = _jpeg((512, 512))  # comfortably over 1 KB
    r = client.post("/v1/predict/image", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413
    assert r.json()["error_code"] == "payload_too_large"
    get_settings.cache_clear()


def test_artifact_path_traversal_is_blocked(client):
    """`..%2F` must not escape the artifact directory."""
    r = client.get("/v1/artifacts/..%2F..%2F..%2Fetc/passwd")
    assert r.status_code in (404, 400)


def test_empty_upload_is_rejected(client):
    r = client.post("/v1/predict/image", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code in (422, 415)


# --------------------------------------------------------------------------- #
# Model info (T54)
# --------------------------------------------------------------------------- #
def test_model_info_carries_the_disclaimer(client):
    body = client.get("/v1/model/info").json()
    assert body["disclaimer"]["not_forensic_evidence"] is True
    assert "image" in body and "device" in body


# --------------------------------------------------------------------------- #
# Jobs (T57)
# --------------------------------------------------------------------------- #
def test_video_without_a_model_is_503(client):
    """The fake registry loads no video model, so video must 503, not 500."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(buf, "PNG")
    # Not a real mp4, but the model check happens before decode.
    r = client.post("/v1/predict/video", files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")})
    assert r.status_code == 503
    assert r.json()["error_code"] == "model_not_ready"


def test_unknown_job_is_404(client):
    r = client.get("/v1/jobs/deadbeefdeadbeef")
    assert r.status_code == 404
    assert r.json()["error_code"] == "job_not_found"
