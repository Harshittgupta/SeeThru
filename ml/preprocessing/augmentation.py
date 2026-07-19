"""Image augmentation pipelines for SEETHRU (albumentations).

Defines two transforms for deepfake-detection training:

* :data:`train_transform` — geometric + photometric augmentation that mimics the
  real-world degradations deepfakes are shared with (re-compression, blur,
  sensor noise, lighting changes), followed by ImageNet normalization.
* :data:`val_transform` — deterministic resize + ImageNet normalization only.

Both are wrapped in :class:`AlbumentationsTransform` so they are *torchvision
compatible*: each is a callable that takes a single PIL image (or HWC numpy
array) and returns a normalized ``CHW`` float tensor — drop-in for the
``transform`` argument of :class:`data.dataset_manager.DeepfakeDataset`.

Pinned to albumentations==1.4.3; the 1.4.x transform API is used throughout.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

# Crop is taken from a slightly larger resize so the random crop has room to
# move; ImageNet normalization stats must match the models' pretrained backbones
# and data.dataset_manager.
#
# RESIZE_SIZE is used by BOTH train and val (T36): train random-crops from it,
# val centre-crops from it. Using it in only one path means the two resample by
# different ratios, which shifts the very spectral distribution the frequency
# branch is trained to read.
RESIZE_SIZE = 256
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _train_aug_list(image_size: int = IMAGE_SIZE) -> list[A.BasicTransform]:
    """The augmentation ops (no normalization/tensor conversion).

    Factored out so :func:`visualize_augmentations` can reuse the exact same
    pipeline and render human-viewable uint8 images.

    **Rotate comes BEFORE RandomCrop, and reflects rather than fills (T37).**

    The original order was Resize -> RandomCrop -> Rotate(border_mode=0), which
    stamped hard black wedges into the corners of the *final* image. Measured on
    200 samples: **41% of training images had black corners**, averaging 1.7% of
    the image area. Validation images never do -- ``border_mode=0`` is
    ``cv2.BORDER_CONSTANT``, i.e. fill with zero.

    Why that is worse than it sounds: a hard edge to black is a step function,
    and a step function is broadband energy in the Fourier domain. The frequency
    branch exists to read exactly that part of the spectrum. So the model was
    being handed a strong, label-independent spectral signal that appears in 41%
    of training images and 0% of eval images -- free capacity spent learning an
    artifact of our own pipeline, and a train/eval distribution shift on top.

    The fix is both halves:
      * ``BORDER_REFLECT_101`` fills from real neighbouring pixels, so no step is
        introduced. (A reflection is not perfectly natural either, but it is
        continuous, and continuity is what the spectrum cares about.)
      * Rotating at 256 and cropping to 224 afterwards means any residual corner
        weirdness is likely to be cropped away rather than baked in.
    """
    return [
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),                # resize to 256×256
        # Rotate first, at the larger size, reflecting the border.
        A.Rotate(
            limit=10,                                      # ±10°, per the spec
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.RandomCrop(image_size, image_size),              # random crop to 224×224
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.6
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.ImageCompression(
            quality_lower=60, quality_upper=95, p=0.5      # quality 60–95
        ),
    ]


# --------------------------------------------------------------------------- #
# torchvision-compatible wrapper
# --------------------------------------------------------------------------- #
class AlbumentationsTransform:
    """Adapt an albumentations ``Compose`` to the torchvision call convention.

    torchvision transforms are called as ``transform(image)`` and operate on a
    single image; albumentations expects ``transform(image=ndarray)['image']``.
    This wrapper bridges the two and accepts either a PIL image or an HWC numpy
    array as input.
    """

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, image):  # noqa: ANN001 - duck-typed PIL/ndarray
        if not isinstance(image, np.ndarray):
            image = np.array(image)  # PIL -> HWC RGB uint8
        return self.transform(image=image)["image"]


def build_train_transform(image_size: int = IMAGE_SIZE) -> AlbumentationsTransform:
    pipeline = A.Compose(
        _train_aug_list(image_size)
        + [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return AlbumentationsTransform(pipeline)


def build_val_transform(image_size: int = IMAGE_SIZE) -> AlbumentationsTransform:
    """Deterministic eval transform: Resize(256) -> CenterCrop(224) -> normalize.

    **The geometry must match training's (T36).** This was previously a bare
    ``Resize(image_size)``, which differs from the train path in two ways at once:

        train: Resize(256) -> RandomCrop(224)  =  87.5% field of view
        val:   Resize(224)                     = 100.0% field of view

    So the model trained on faces at one zoom level and was evaluated on another
    -- and, less obviously but more damagingly, the two paths resample by
    *different ratios* (e.g. 300->256 vs 300->224). Resampling leaves its own
    signature in the frequency domain, and the frequency branch is built to read
    frequency-domain signatures. Evaluating with a different resampling ratio
    than training used shifts the exact distribution that branch learned.

    Resize-then-CenterCrop reproduces the train path's zoom and resample ratio
    while staying deterministic (the crop is centred, not random).
    """
    pipeline = A.Compose(
        [
            A.Resize(RESIZE_SIZE, RESIZE_SIZE),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return AlbumentationsTransform(pipeline)


# Ready-to-use default instances.
train_transform = build_train_transform()
val_transform = build_val_transform()


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def visualize_augmentations(image_path: str, n: int = 8, seed: int = 0) -> None:
    """Show a grid of ``n`` augmented versions of one image with matplotlib.

    Renders the *un-normalized* augmentations (uint8) so the result is directly
    viewable; this uses the same augmentation ops as :data:`train_transform`.

    Note: matplotlib is only needed for this helper and is imported lazily.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    image = np.array(Image.open(image_path).convert("RGB"))

    # Same augmentation ops as training, but without Normalize/ToTensor so the
    # output stays a displayable uint8 image.
    display_pipeline = A.Compose(_train_aug_list(IMAGE_SIZE))

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            aug = display_pipeline(image=image)["image"]
            ax.imshow(aug)
            ax.set_title(f"aug {i + 1}", fontsize=9)

    fig.suptitle("Training augmentations", fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview training augmentations.")
    parser.add_argument("image", help="Path to a sample image.")
    parser.add_argument("-n", type=int, default=8, help="Number of variants.")
    args = parser.parse_args()

    # Quick sanity report on tensor output shape.
    from PIL import Image

    sample = Image.open(args.image).convert("RGB")
    t = train_transform(sample)
    v = val_transform(sample)
    print(f"train_transform output: shape={tuple(t.shape)}, dtype={t.dtype}")
    print(f"val_transform   output: shape={tuple(v.shape)}, dtype={v.dtype}")

    visualize_augmentations(args.image, n=args.n)
