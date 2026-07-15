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

from typing import List

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

# Crop is taken from a slightly larger resize so the random crop has room to
# move; ImageNet normalization stats must match the models' pretrained backbones
# and data.dataset_manager.
RESIZE_SIZE = 256
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _train_aug_list(image_size: int = IMAGE_SIZE) -> List[A.BasicTransform]:
    """The augmentation ops (no normalization/tensor conversion).

    Factored out so :func:`visualize_augmentations` can reuse the exact same
    pipeline and render human-viewable uint8 images.
    """
    return [
        A.Resize(RESIZE_SIZE, RESIZE_SIZE),                # resize to 256×256
        A.RandomCrop(image_size, image_size),              # random crop to 224×224
        A.Rotate(limit=10, border_mode=0, p=0.7),          # ±10°
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
    pipeline = A.Compose(
        [
            A.Resize(image_size, image_size),
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
