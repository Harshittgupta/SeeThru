"""Shared pytest fixtures for SEETHRU.

The governing constraint here: **fixtures must be generated, never committed.**

``data/dummy/`` is gitignored (and the real datasets are EULA-restricted, so they
can never be committed at all), which means CI checks out a repo with zero test
media in it. Any fixture that reads from ``data/`` would pass locally and fail in
CI. So every fixture below builds what it needs into ``tmp_path``.

Second constraint: **nothing here may need a GPU, network, or trained weights.**
Models are always constructed with ``pretrained=False`` so torchvision never
reaches out to download ImageNet weights.

See docs/BUILD_PLAN.md T9.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Constants mirrored from the production code. Kept local on purpose: if someone
# changes IMAGE_SIZE in ml/, these tests should fail loudly rather than silently
# following along.
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root (tests/conftest.py -> SEETHRU/)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded RNG. Seeded so a failure is reproducible from the traceback."""
    return np.random.default_rng(1234)


# --------------------------------------------------------------------------- #
# Synthetic media
# --------------------------------------------------------------------------- #
@pytest.fixture
def random_image(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    """Factory for a random HWC uint8 BGR image (OpenCV's channel order)."""

    def _make(height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
        return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    return _make


@pytest.fixture
def tiny_jpeg_bytes(rng: np.random.Generator) -> bytes:
    """An 8x8 JPEG as raw bytes.

    Deliberately tiny: upload-validation tests (T55) care about magic bytes and
    size limits, not pixels, and an 8x8 keeps the suite fast.
    """
    from PIL import Image

    arr = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> Path:
    """Generate the dummy image/video dataset into ``tmp_path``.

    Reuses ml/preprocessing/create_dummy_dataset.py so the fixture and the real
    generator can never drift apart. Returns the ``data/dummy`` root.

    NOTE: as of T9 this generator emits ``person_001_frame_001.jpg`` naming,
    which does NOT exercise the real FF++/Celeb-DF identity traps. T12 fixes the
    generator to emit ``033_097.mp4`` / ``id3_id5_0001.mp4`` shapes; this fixture
    picks that up for free when it lands.
    """
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ml" / "preprocessing"))
    from create_dummy_dataset import generate_images  # noqa: PLC0415

    images_root = tmp_path / "data" / "dummy" / "images"
    generate_images(images_root, np.random.default_rng(42))
    return tmp_path / "data" / "dummy"


@pytest.fixture
def dummy_images_root(dummy_dataset: Path) -> Path:
    """The ``real/``+``fake/`` image root that DeepfakeDataset expects."""
    return dummy_dataset / "images"


@pytest.fixture(scope="session")
def _dataset_shaped_trees(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped: miniature FF++ and Celeb-DF trees.

    Session-scoped because encoding ~60 mp4s costs a few seconds and none of the
    consumers mutate them. Returns the ``data/dummy`` root containing both
    ``ffpp_like/`` and ``celebdf_like/``.
    """
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ml" / "preprocessing"))
    from create_dummy_dataset import (  # noqa: PLC0415
        generate_celebdf_like,
        generate_ffpp_like,
    )

    base = tmp_path_factory.mktemp("datasets")
    rng = np.random.default_rng(42)
    generate_ffpp_like(base / "ffpp_like", rng)
    generate_celebdf_like(base / "celebdf_like", rng)
    return base


@pytest.fixture
def ffpp_root(_dataset_shaped_trees: Path) -> Path:
    """A miniature FaceForensics++ tree, real naming and all.

    Contains reciprocal swap pairs (000_001 AND 001_000) so the two-identity
    leak of T15 is reproducible offline, plus official splits/*.json.
    """
    return _dataset_shaped_trees / "ffpp_like"


@pytest.fixture
def celebdf_root(_dataset_shaped_trees: Path) -> Path:
    """A miniature Celeb-DF v2 tree, incl. List_of_testing_videos.txt."""
    return _dataset_shaped_trees / "celebdf_like"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@pytest.fixture
def image_model():
    """An untrained ImageClassifier on CPU.

    ``pretrained=False`` is load-bearing: it keeps the suite offline (no
    torchvision weight download) and fast. Tests here assert shapes and
    plumbing, which random weights exercise perfectly well.

    ⚠️ **KNOWN LIMITATION -- read before writing a test that depends on the
    spatial branch's VALUES.**

    ``SpatialBranch(pretrained=False).eval()`` outputs **effectively zero**
    (measured std ``7.4e-15``, vs ``8.0e-02`` in ``train()`` mode and
    ``8.7e-02`` with real ImageNet weights). EfficientNet's BatchNorms carry
    their *initial* running stats (mean=0, var=1), so in eval mode they act as
    the identity, nothing rescales the signal between layers, and it collapses
    on its way through the network.

    Consequences:

    * Shape/plumbing tests are still perfectly valid -- a zero vector has a
      shape, and the wiring is what they check.
    * But **1536 of the fusion MLP's 2176 inputs are identically zero here**, so
      any test asserting that the spatial branch *influences* an output will pass
      vacuously and prove nothing.
    * In particular, ablation attribution (T51) must NOT be tested against this
      fixture: ablating an already-zero branch produces a delta of exactly 0.0,
      which looks like a working test and is not. Use
      :func:`synthetic_branch_features` (fast, offline, isolates fusion+head --
      the part ablation actually exercises), or ``pretrained=True`` behind
      ``@pytest.mark.slow``.
    """
    import torch

    from ml.models.classifier import ImageClassifier

    torch.manual_seed(0)
    model = ImageClassifier(pretrained=False)
    model.eval()
    return model


@pytest.fixture
def synthetic_branch_features():
    """Realistic per-branch feature tensors, without running a backbone.

    Exists because of the limitation documented on :func:`image_model`: an
    untrained backbone in eval mode emits zeros, so tests that need the branches
    to actually *carry signal* have to synthesise it.

    Fast (no backbone), offline (no weight download), and it isolates fusion +
    head -- which is exactly the surface ablation attribution operates on (T51).
    Returns ``(spatial, frequency, temporal)`` for a batch of 4.
    """
    import torch

    from ml.models.fusion import FREQUENCY_DIM, SPATIAL_DIM, TEMPORAL_DIM

    torch.manual_seed(0)
    return (
        # Post-ReLU/pool features are non-negative, so match that rather than
        # using randn -- feeding the MLP a distribution it never sees in
        # training would make any measured effect an artifact of the fixture.
        torch.rand(4, SPATIAL_DIM),
        torch.rand(4, FREQUENCY_DIM),
        torch.rand(4, TEMPORAL_DIM),
    )


@pytest.fixture
def video_model():
    """An untrained VideoClassifier on CPU. See `image_model` re: pretrained."""
    import torch

    from ml.models.classifier import VideoClassifier

    torch.manual_seed(0)
    model = VideoClassifier(pretrained=False)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip @pytest.mark.gpu tests when no CUDA device is present.

    They are already deselected by the default `-m 'not gpu'` in pyproject.toml;
    this is the backstop for someone running `pytest -m gpu` on a CPU box, so
    they get a clear skip instead of a confusing CUDA error.
    """
    if any(mark.name == "gpu" for mark in item.iter_markers()):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("requires CUDA; none available on this machine")
