"""Augmentation pipeline tests (BUILD_PLAN T14) + the albumentations API canary (T4).

The canary is the point of this file. `albumentations==1.4.3` is a LOAD-BEARING
pin: augmentation.py calls `GaussNoise(var_limit=...)` and
`ImageCompression(quality_lower=..., quality_upper=...)`, and all three kwargs
were renamed or removed in >=1.4.14 (`std_range`, `quality_range`). A dependabot
PR bumping the pin would break training augmentation -- and, because
albumentations resolves kwargs loosely in some versions, could break it *quietly*,
producing a model trained without the compression robustness the whole frequency
branch depends on.

A comment in requirements.txt does not stop that. A failing test does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.preprocessing.augmentation import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_train_transform,
    build_val_transform,
)


# --------------------------------------------------------------------------- #
# The canary (T4)
# --------------------------------------------------------------------------- #
def test_albumentations_pin_is_still_valid():
    """The exact kwargs augmentation.py depends on must still construct.

    If this fails, someone bumped albumentations past 1.4.13. Do not "fix" it by
    deleting the test: port augmentation.py to the new API (var_limit ->
    std_range, quality_lower/upper -> quality_range) and update this canary.
    """
    import albumentations as A

    # Removed/renamed in >=1.4.14.
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
    A.ImageCompression(quality_lower=60, quality_upper=95, p=1.0)


def test_albumentations_version_is_pinned():
    """Belt and braces: the pin itself, so the failure names the cause."""
    import albumentations as A

    assert A.__version__ == "1.4.3", (
        f"albumentations is {A.__version__}, expected 1.4.3. The pin is "
        f"load-bearing -- see requirements.txt and augmentation.py:50-53."
    )


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("build", [build_train_transform, build_val_transform])
def test_transform_output_contract(build, random_image):
    """CHW float32 at the model's expected size -- what the backbone requires."""
    transform = build()
    out = transform(random_image(480, 640))  # non-square input, deliberately

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("build", [build_train_transform, build_val_transform])
def test_transform_accepts_pil_and_numpy(build, random_image):
    """DeepfakeDataset hands these a PIL image; video code hands them ndarrays."""
    from PIL import Image

    transform = build()
    arr = random_image()
    from_numpy = transform(arr)
    from_pil = transform(Image.fromarray(arr))

    assert from_numpy.shape == from_pil.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_val_transform_is_deterministic(random_image):
    """Val must not augment. A random val transform makes the metric noise."""
    transform = build_val_transform()
    img = random_image()
    assert torch.allclose(transform(img), transform(img))


def test_train_transform_actually_augments(random_image):
    """...and train must. An augmentation pipeline that returns the same tensor
    twice is a pipeline whose p= values are all zero."""
    transform = build_train_transform()
    img = random_image()
    outs = [transform(img) for _ in range(6)]
    assert any(not torch.allclose(outs[0], o) for o in outs[1:]), (
        "train_transform produced identical output 6 times -- is it augmenting?"
    )


# --------------------------------------------------------------------------- #
# Geometry: train and val must match (T36/T37)
# --------------------------------------------------------------------------- #
def test_train_never_stamps_black_borders(random_image):
    """Rotation must reflect, not fill with black (T37).

    Before the fix: `Resize -> RandomCrop -> Rotate(border_mode=0)` put hard
    black wedges in the corners of 41% of training images (measured over 200
    samples, averaging 1.7% of the image area), while val images never had them.

    A hard edge to black is a step function, and a step function is broadband
    energy in the Fourier domain -- which is precisely the part of the spectrum
    the frequency branch reads. The model was being handed a strong,
    label-independent spectral cue present in 41% of train images and 0% of eval
    images.
    """
    import albumentations as A

    from ml.preprocessing.augmentation import _train_aug_list

    pipeline = A.Compose(_train_aug_list())
    # Mid-grey: any exact zero in the output can only have come from a border fill.
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)

    black_fractions = [
        (pipeline(image=flat)["image"] == 0).all(axis=2).mean() for _ in range(60)
    ]
    worst = max(black_fractions)
    assert worst < 0.001, (
        f"{sum(f > 0.001 for f in black_fractions)}/60 augmented images contain "
        f"black border fill (worst covers {worst:.1%} of the image). Is Rotate "
        f"using border_mode=0, or running after RandomCrop again?"
    )


def test_train_and_val_share_field_of_view():
    """Train and val must crop the same fraction of the image (T36).

    train: Resize(256) -> RandomCrop(224)  = 87.5% FOV
    val:   Resize(224)                     = 100% FOV   <- the bug

    Different zoom AND a different resampling ratio. Resampling leaves a
    frequency-domain signature, and the frequency branch is built to read those,
    so a val path that resamples differently shifts the distribution out from
    under it.
    """
    from ml.preprocessing.augmentation import IMAGE_SIZE as SIZE
    from ml.preprocessing.augmentation import RESIZE_SIZE, build_val_transform

    ops = {type(t).__name__ for t in build_val_transform().transform.transforms}
    assert "CenterCrop" in ops, (
        "val must Resize-then-CenterCrop to match train's field of view and "
        "resample ratio, not bare-Resize to the final size"
    )

    resize = next(
        t for t in build_val_transform().transform.transforms
        if type(t).__name__ == "Resize"
    )
    assert (resize.height, resize.width) == (RESIZE_SIZE, RESIZE_SIZE), (
        f"val resizes to {resize.height}, train resizes to {RESIZE_SIZE} -- "
        f"different resample ratios"
    )
    crop = next(
        t for t in build_val_transform().transform.transforms
        if type(t).__name__ == "CenterCrop"
    )
    assert (crop.height, crop.width) == (SIZE, SIZE)


def test_val_crop_is_centred_and_deterministic(random_image):
    """CenterCrop, not RandomCrop -- eval must not be stochastic."""
    transform = build_val_transform()
    img = random_image(300, 300)
    assert torch.allclose(transform(img), transform(img))


def test_normalization_uses_imagenet_stats(random_image):
    """Stats must match the pretrained backbone's, or transfer learning starts
    from a distribution the weights have never seen."""
    transform = build_val_transform()
    # A constant image lets us invert the normalization exactly.
    flat = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    out = transform(flat)

    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        expected = ((128 / 255.0) - mean) / std
        assert out[channel].mean().item() == pytest.approx(expected, abs=1e-3), (
            f"channel {channel} normalization does not match ImageNet stats"
        )
