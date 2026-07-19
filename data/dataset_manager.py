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
from collections.abc import Callable, Sequence
from pathlib import Path

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

# A split needs at least one identity in each of train/val/test.
MIN_IDENTITIES = 3

# Below this many samples, "one identity per file" is plausible rather than a
# symptom of a broken identity_fn, so we don't flag it.
_EXPLOSION_CHECK_MIN_SAMPLES = 10

# Matches an explicit trailing frame index: "_frame_001", "-frame-7", "_frame7".
_FRAME_SUFFIX_RE = re.compile(r"[_-]frame[_-]?\d+$", re.IGNORECASE)


def default_identity_fn(path: Path) -> str:
    """Derive a subject id from a filename with simple ``<identity>[_frame_<n>]`` naming.

    Strips an explicit trailing frame index and returns the rest of the stem::

        person_001_frame_001.jpg  ->  person_001
        person_001.mp4            ->  person_001   (a subject's video and its
                                                    frames map to the same id)

    **This heuristic cannot handle the real datasets, and does not try to.**
    FaceForensics++ and Celeb-DF encode *two* identities in every fake filename
    (``033_097`` = target 033 with source 097's face swapped in; ``id3_id5_0001``
    likewise). Grouping on either id alone leaks the other across splits -- the
    trap that BUILD_PLAN T15 exists to close. Use the dataset-specific functions
    for those, or better, FF++'s official ``splits/*.json``.

    A previous version of this function took the leading alphanumeric run
    (``^([A-Za-z0-9]+)``), which mapped every ``person_XXX_frame_YYY`` to the
    single identity ``"person"``. With one identity, ``round(0.15 * 1) == 0``, so
    val and test came back **empty with no error raised**. Wrong guesses are now
    caught at construction by :meth:`DeepfakeDataset._validate_identities`
    instead of being silently tolerated.
    """
    stem = path.stem
    stripped = _FRAME_SUFFIX_RE.sub("", stem)
    # Never return "" -- a file literally named "frame_001.jpg" would otherwise
    # produce an empty identity that silently groups with every other such file.
    return stripped or stem


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
        validate_identities: If ``True`` (default), raise when ``identity_fn``
            has obviously misfired -- see :meth:`_validate_identities`. Set
            ``False`` only if your dataset genuinely has one image per subject,
            which makes the "explosion" check a false positive.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        balance: bool = True,
        image_size: int = 224,
        transform: Callable | None = None,
        identity_fn: Callable[[Path], str] | None = None,
        seed: int = 42,
        validate_identities: bool = True,
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

        # Catch a broken identity_fn BEFORE splitting. Both failure directions
        # are silent if unchecked, and both invalidate every downstream metric.
        if validate_identities:
            self._validate_identities(all_samples)

        # Assign identities to splits, then keep only samples for this split.
        if split == "all":
            split_samples = all_samples
        else:
            identity_to_split = self._assign_identity_splits(all_samples)
            split_samples = [
                s for s in all_samples if identity_to_split[s[2]] == split
            ]
            self._require_non_empty_split(split_samples, all_samples)

        # Enforce 50:50 class balance within the split.
        if self.balance:
            split_samples = self._balance_classes(split_samples)

        # Final ordering is deterministic (sorted by path) for reproducibility.
        self.samples: list[tuple[Path, int, str]] = sorted(
            split_samples, key=lambda s: str(s[0])
        )

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return ``(tensor, label)``.

        Kept as a plain tuple: this is the conventional torchvision contract and
        every training loop expects to unpack two values. Per-sample metadata
        (identity, path) is available via :meth:`metadata` or ``self.samples``,
        which is enough for the metrics that need it -- unlike the video dataset,
        the image set has no ``manipulation`` field to lose (T19).
        """
        path, label, _identity = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
            tensor = self.transform(image)
        return tensor, label

    def metadata(self, index: int) -> dict[str, object]:
        """Per-sample metadata for the sample at ``index``.

        Lets an evaluation loop group predictions by identity without paying to
        decode the image, and without changing __getitem__'s tuple contract.
        """
        path, label, identity = self.samples[index]
        return {
            "path": str(path),
            "label": label,
            "identity": identity,
            "class_name": LABEL_TO_CLASS[label],
        }

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def class_counts(self) -> dict[str, int]:
        """Return the number of samples per class in this split."""
        counts = {name: 0 for name in CLASS_TO_LABEL}
        for _path, label, _identity in self.samples:
            counts[LABEL_TO_CLASS[label]] += 1
        return counts

    def identities(self) -> list[str]:
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

    def _validate_identities(self, samples: Sequence[tuple[Path, int, str]]) -> None:
        """Fail loudly when ``identity_fn`` has obviously misfired.

        Identity-separated splitting has two silent failure modes, and both
        produce a dataset that looks fine and reports numbers that are fiction:

        * **Collapse** -- every file maps to the same id. There is 1 identity,
          ``round(0.15 * 1) == 0``, and val/test are empty. Nothing raises, so
          early stopping on val loss never fires and you find out at write-up.
        * **Explosion** -- every file maps to a *unique* id. "Identity-separated"
          silently degenerates into a random per-sample split, the same subject
          lands in train and test, and metrics inflate.

        Neither is detectable from ``len(dataset)`` alone, which is why this runs
        at construction rather than being left to the caller.
        """
        identities = {identity for _p, _l, identity in samples}
        n_ids = len(identities)
        n_samples = len(samples)

        def _example_mapping(limit: int = 3) -> str:
            rows = [f"{p.name!r} -> {identity!r}" for p, _l, identity in samples[:limit]]
            return "; ".join(rows)

        if n_ids < MIN_IDENTITIES:
            raise ValueError(
                f"identity_fn produced only {n_ids} distinct "
                f"{'identity' if n_ids == 1 else 'identities'} across {n_samples} "
                f"samples under {self.root}, but at least {MIN_IDENTITIES} are "
                f"needed to fill train/val/test.\n"
                f"  Got: {sorted(identities)[:5]}\n"
                f"  Example mapping: {_example_mapping()}\n"
                f"  This collapses the split: val/test would be EMPTY.\n"
                f"  Fix the filenames, or pass an identity_fn matching your "
                f"dataset's naming."
            )

        if n_samples >= _EXPLOSION_CHECK_MIN_SAMPLES and n_ids == n_samples:
            raise ValueError(
                f"identity_fn produced a unique identity for every one of the "
                f"{n_samples} samples under {self.root}, so nothing is grouped "
                f"and the identity split degenerates into a random one -- frames "
                f"of the same subject will land in both train and test.\n"
                f"  Example mapping: {_example_mapping()}\n"
                f"  Pass an identity_fn that extracts the SUBJECT from the "
                f"filename, not the file's own name.\n"
                f"  (If your dataset genuinely has exactly one image per subject, "
                f"this warning is a false positive -- identity splitting is then "
                f"equivalent to random splitting and is harmless. Pass "
                f"validate_identities=False to proceed.)"
            )

    def _require_non_empty_split(
        self,
        split_samples: Sequence[tuple[Path, int, str]],
        all_samples: Sequence[tuple[Path, int, str]],
    ) -> None:
        """Refuse to hand back an empty split.

        Returning ``len == 0`` here is the single most expensive failure mode in
        this codebase: a DataLoader over an empty dataset yields no batches, the
        val loop reports nothing, and training "succeeds" having never validated.
        """
        if split_samples:
            return

        n_ids = len({identity for _p, _l, identity in all_samples})
        train_r, val_r, test_r = self.split_ratios
        raise ValueError(
            f"split {self.split!r} is empty: {n_ids} identities split by "
            f"{self.split_ratios} leaves nothing for it "
            f"(n_train={int(round(train_r * n_ids))}, "
            f"n_val={int(round(val_r * n_ids))}, "
            f"n_test={n_ids - int(round(train_r * n_ids)) - int(round(val_r * n_ids))} "
            f"identities).\n"
            f"  Either supply more identities, or widen split_ratios."
        )

    def _scan_samples(self) -> list[tuple[Path, int, str]]:
        """Walk ``real/`` and ``fake/`` and collect (path, label, identity)."""
        samples: list[tuple[Path, int, str]] = []
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
        self, samples: Sequence[tuple[Path, int, str]]
    ) -> dict[str, str]:
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

        assignment: dict[str, str] = {}
        for i, identity in enumerate(shuffled):
            if i < n_train:
                assignment[identity] = "train"
            elif i < n_train + n_val:
                assignment[identity] = "val"
            else:
                assignment[identity] = "test"
        return assignment

    def _balance_classes(
        self, samples: Sequence[tuple[Path, int, str]]
    ) -> list[tuple[Path, int, str]]:
        """Downsample the majority class to enforce a 50:50 real/fake ratio."""
        by_label: dict[int, list[tuple[Path, int, str]]] = defaultdict(list)
        for sample in samples:
            by_label[sample[1]].append(sample)

        if not by_label:
            return list(samples)

        # If a class is entirely missing, nothing to balance against.
        if len(by_label) < len(CLASS_TO_LABEL):
            return list(samples)

        target = min(len(items) for items in by_label.values())

        generator = torch.Generator().manual_seed(self.seed)
        balanced: list[tuple[Path, int, str]] = []
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
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    **kwargs,
) -> dict[str, DeepfakeDataset]:
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
