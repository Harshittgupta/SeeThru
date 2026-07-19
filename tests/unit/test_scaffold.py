"""Validates the test scaffold itself (BUILD_PLAN T9).

These are not tests of SEETHRU. They are tests that the *harness* works: that
fixtures generate, that models import and run offline on CPU, and that the gpu
marker deselects. If these fail, every other test's result is meaningless.

The real tests start at T13 (identity leakage) and T14 (splits, transforms,
shapes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_random_image_fixture(random_image):
    """The image factory returns HWC uint8 in OpenCV's expected shape."""
    img = random_image()
    assert img.shape == (224, 224, 3)
    assert img.dtype == np.uint8


def test_tiny_jpeg_has_jpeg_magic_bytes(tiny_jpeg_bytes):
    """The JPEG fixture is a real JPEG.

    Matters because T55's upload validation rejects on magic bytes, so the
    fixture must carry genuine ones (0xFF 0xD8 0xFF) rather than just a name.
    """
    assert tiny_jpeg_bytes[:3] == b"\xff\xd8\xff"
    assert len(tiny_jpeg_bytes) > 0


def test_dummy_dataset_generates_into_tmp_path(dummy_images_root: Path, tmp_path: Path):
    """Fixtures are generated, never read from the repo.

    This is the property that keeps CI honest: data/dummy/ is gitignored, so a
    fixture reading from it would pass here and fail on a fresh checkout.
    """
    assert tmp_path in dummy_images_root.parents

    real = list((dummy_images_root / "real").glob("*.jpg"))
    fake = list((dummy_images_root / "fake").glob("*.jpg"))
    assert len(real) == 40
    assert len(fake) == 40
    assert len(real) == len(fake), "dummy set must be 50:50"


def test_image_model_forward_offline_on_cpu(image_model):
    """An untrained model imports, runs on CPU, and emits (B, 2) logits.

    pretrained=False keeps this offline -- no torchvision weight download.
    """
    import torch

    logits = image_model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.gpu
def test_gpu_marker_is_deselected_by_default():
    """Should never run under a bare `pytest`.

    If this executes in CI, the -m 'not gpu' default in pyproject.toml has
    broken and GPU tests will start failing on GPU-less runners.
    """
    import torch

    assert torch.cuda.is_available()
