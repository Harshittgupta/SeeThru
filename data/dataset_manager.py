"""Dataset management for SEETHRU deepfake detection.

Provides :class:`DeepfakeDataset`, a ``torch.utils.data.Dataset`` that loads
real/fake face images from a root directory laid out as::

    root/
    ├── real/   # genuine faces
    └── fake/   # manipulated / synthetic faces

Key features:

* **Class balance** — enforces a 50:50 real/fake ratio within each split by
  downsampling the majority class (seeded, reproducible).
* **Identity-separated splits** — train/val/test are split by *subject* so that
  no identity appears in more than one split (prevents identity leakage, which
  would inflate evaluation metrics).
* **ImageNet normalization** — images are returned as tensors normalized to the
  standard ImageNet mean/std, ready for pretrained backbones.

The split for every ``DeepfakeDataset`` instance is computed deterministically
from ``(root, seed, split_ratios)``, so constructing the ``"train"``, ``"val"``
and ``"test"`` datasets separately yields a consistent, non-overlapping
partition.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Standard ImageNet normalization statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Image file extensions we recognise.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class folder names -> integer labels (0 = real, 1 = fake).
CLASS_TO_LABEL = {"real": 0, "fake": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

VALID_SPLITS = ("train", "val", "test", "all")


def default_identity_fn(path: Path) -> str:
    """Derive a subject/identity id from an image file path.

    Many deepfake datasets encode the subject in the filename (e.g.
    ``id3_id5_0001.png`` in Celeb-DF, or ``033_097.png`` in FaceForensics++).
    The default heuristic takes the leading token before the first separator
    (``_`` or ``-``); if no separator is present the full stem is used.

    Override this with the ``identity_fn`` constructor argument to match your
    dataset's naming convention.
    """
    stem = path.stem
    match = re.match(r"^([A-Za-z0-9]+)", stem)
    token = match.group(1) if match else stem
    return token


class DeepfakeDataset(Dataset):
    """Real/fake face dataset with balanced, identity-separated splits.

    Args:
        root: Directory containing ``real/`` and ``fake/`` subfolders.
        split: One of ``"train"``, ``"val"``, ``"test"`` or ``"all"``.
        split_ratios: ``(train, val, test)`` fractions; must sum to ~1.0.
            Ignored when ``split == "all"``.
        balance: If ``True`` (default) enforce a 50:50 real/fake ratio in this
            split by randomly downsampling the majority class.
        image_size: Output spatial size (images are resized to a square).
        transform: Optional custom transform applied to a PIL image. If
            ``None``, a default ``Resize -> ToTensor -> Normalize(ImageNet)``
            pipeline is used.
        identity_fn: Callable mapping an image ``Path`` to an identity string,
            used for identity-separated splitting. Defaults to
            :func:`default_identity_fn`.
        seed: Seed controlling the (deterministic) identity split and balancing.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        balance: bool = True,
        image_size: int = 224,
        transform: Optional[Callable] = None,
        identity_fn: Optional[Callable[[Path], str]] = None,
        seed: int = 42,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.split = split
        self.split_ratios = split_ratios
        self.balance = balance
        self.image_size = image_size
        self.identity_fn = identity_fn or default_identity_fn
        self.seed = seed

        self.transform = transform or self._default_transform(image_size)

        # Scan disk -> list of (path, label, identity) for every image.
        all_samples = self._scan_samples()
        if not all_samples:
            raise RuntimeError(
                f"No images found under {self.root}/real and {self.root}/fake"
            )

        # Assign identities to splits, then keep only samples for this split.
        if split == "all":
            split_samples = all_samples
        else:
            identity_to_split = self._assign_identity_splits(all_samples)
            split_samples = [
                s for s in all_samples if identity_to_split[s[2]] == split
            ]

        # Enforce 50:50 class balance within the split.
        if self.balance:
            split_samples = self._balance_classes(split_samples)

        # Final ordering is deterministic (sorted by path) for reproducibility.
        self.samples: List[Tuple[Path, int, str]] = sorted(
            split_samples, key=lambda s: str(s[0])
        )

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label, _identity = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
            tensor = self.transform(image)
        return tensor, label

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def class_counts(self) -> Dict[str, int]:
        """Return the number of samples per class in this split."""
        counts = {name: 0 for name in CLASS_TO_LABEL}
        for _path, label, _identity in self.samples:
            counts[LABEL_TO_CLASS[label]] += 1
        return counts

    def identities(self) -> List[str]:
        """Return the sorted unique identities present in this split."""
        return sorted({identity for _p, _l, identity in self.samples})

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_transform(image_size: int) -> Callable:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _scan_samples(self) -> List[Tuple[Path, int, str]]:
        """Walk ``real/`` and ``fake/`` and collect (path, label, identity)."""
        samples: List[Tuple[Path, int, str]] = []
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class subfolder not found: {class_dir}"
                )
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    identity = self.identity_fn(path)
                    samples.append((path, label, identity))
        return samples

    def _assign_identity_splits(
        self, samples: Sequence[Tuple[Path, int, str]]
    ) -> Dict[str, str]:
        """Deterministically map each identity to a train/val/test split.

        Splitting is done at the identity level (not the sample level) so that
        every image of a subject lands in the same split — no subject can appear
        in more than one split.
        """
        train_r, val_r, test_r = self.split_ratios
        total = train_r + val_r + test_r
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {total}")

        identities = sorted({identity for _p, _l, identity in samples})

        # Seeded deterministic shuffle of identities.
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(identities), generator=generator).tolist()
        shuffled = [identities[i] for i in perm]

        n = len(shuffled)
        n_train = int(round(train_r * n))
        n_val = int(round(val_r * n))
        # Remainder goes to test to absorb rounding.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        assignment: Dict[str, str] = {}
        for i, identity in enumerate(shuffled):
            if i < n_train:
                assignment[identity] = "train"
            elif i < n_train + n_val:
                assignment[identity] = "val"
            else:
                assignment[identity] = "test"
        return assignment

    def _balance_classes(
        self, samples: Sequence[Tuple[Path, int, str]]
    ) -> List[Tuple[Path, int, str]]:
        """Downsample the majority class to enforce a 50:50 real/fake ratio."""
        by_label: Dict[int, List[Tuple[Path, int, str]]] = defaultdict(list)
        for sample in samples:
            by_label[sample[1]].append(sample)

        if not by_label:
            return list(samples)

        # If a class is entirely missing, nothing to balance against.
        if len(by_label) < len(CLASS_TO_LABEL):
            return list(samples)

        target = min(len(items) for items in by_label.values())

        generator = torch.Generator().manual_seed(self.seed)
        balanced: List[Tuple[Path, int, str]] = []
        for label in sorted(by_label):
            items = sorted(by_label[label], key=lambda s: str(s[0]))
            if len(items) > target:
                perm = torch.randperm(len(items), generator=generator).tolist()
                keep = sorted(perm[:target])
                items = [items[i] for i in keep]
            balanced.extend(items)
        return balanced


def build_splits(
    root: str | Path,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    **kwargs,
) -> Dict[str, DeepfakeDataset]:
    """Convenience constructor returning the three identity-separated splits.

    Returns a dict ``{"train": ..., "val": ..., "test": ...}``.
    """
    return {
        split: DeepfakeDataset(
            root, split=split, split_ratios=split_ratios, **kwargs
        )
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a DeepfakeDataset.")
    parser.add_argument("root", help="Dataset root containing real/ and fake/")
    parser.add_argument("--split", default="train", choices=VALID_SPLITS)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = DeepfakeDataset(
        args.root, split=args.split, image_size=args.image_size, seed=args.seed
    )
    print(f"Split '{args.split}': {len(ds)} samples")
    print(f"Class counts: {ds.class_counts()}")
    print(f"Unique identities: {len(ds.identities())}")
    if len(ds):
        tensor, label = ds[0]
        print(f"Sample 0: tensor {tuple(tensor.shape)}, label {label}")
