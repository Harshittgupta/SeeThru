"""Generate a synthetic dummy dataset for SEETHRU pipeline testing.

Creates random-noise images and videos in the on-disk layout the real dataset
uses, so the full preprocessing/training/inference pipeline can be smoke-tested
without any real data (or any dependency beyond numpy and OpenCV).

Layout produced (under ``<root>/data/dummy``)::

    images/real/person_001_frame_001.jpg ... person_010_frame_004.jpg   (40)
    images/fake/person_001_frame_001.jpg ... person_010_frame_004.jpg   (40)
    videos/real/person_001.mp4         ... person_010.mp4               (10)
    videos/fake/person_001.mp4         ... person_010.mp4               (10)

Identities ``person_001``..``person_010`` each contribute 4 image frames and one
video, matching the identity-aware splitting in data/dataset_manager.py.

Usage::

    python ml/preprocessing/create_dummy_dataset.py
    python ml/preprocessing/create_dummy_dataset.py --root /tmp/seethru --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Dataset shape constants.
IMG_SIZE = 224
NUM_IDENTITIES = 10
FRAMES_PER_IDENTITY = 4          # -> 40 images per class
VIDEO_FRAMES = 16
VIDEO_FPS = 30
CLASSES = ("real", "fake")


def _repo_root() -> Path:
    """Repository root: ml/preprocessing/<file> -> SEETHRU/."""
    return Path(__file__).resolve().parents[2]


def _random_image(rng: np.random.Generator) -> np.ndarray:
    """A random 224×224×3 uint8 BGR image."""
    return rng.integers(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def generate_images(images_root: Path, rng: np.random.Generator) -> int:
    """Write the per-class image frames. Returns total images written."""
    written = 0
    for class_name in CLASSES:
        class_dir = images_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for pid in range(1, NUM_IDENTITIES + 1):
            for frame in range(1, FRAMES_PER_IDENTITY + 1):
                fname = f"person_{pid:03d}_frame_{frame:03d}.jpg"
                ok = cv2.imwrite(str(class_dir / fname), _random_image(rng))
                if not ok:
                    raise RuntimeError(f"Failed to write image {fname}")
                written += 1
    return written


def generate_videos(videos_root: Path, rng: np.random.Generator) -> int:
    """Write the per-class dummy videos. Returns total videos written."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    written = 0
    for class_name in CLASSES:
        class_dir = videos_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for pid in range(1, NUM_IDENTITIES + 1):
            fname = f"person_{pid:03d}.mp4"
            path = class_dir / fname
            writer = cv2.VideoWriter(
                str(path), fourcc, VIDEO_FPS, (IMG_SIZE, IMG_SIZE)
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"VideoWriter failed to open for {path} "
                    "(is the mp4v codec available?)"
                )
            try:
                for _ in range(VIDEO_FRAMES):
                    writer.write(_random_image(rng))
            finally:
                writer.release()
            written += 1
    return written


def _dir_stats(directory: Path) -> Tuple[int, int]:
    """Return (file_count, total_bytes) for files directly in ``directory``."""
    count = 0
    size = 0
    if directory.is_dir():
        for f in directory.iterdir():
            if f.is_file():
                count += 1
                size += f.stat().st_size
    return count, size


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def print_summary(dummy_root: Path) -> None:
    """Print a table of folder paths, file counts and on-disk size."""
    folders: List[Path] = [
        dummy_root / "images" / "real",
        dummy_root / "images" / "fake",
        dummy_root / "videos" / "real",
        dummy_root / "videos" / "fake",
    ]

    rows = [(str(f), *_dir_stats(f)) for f in folders]
    path_width = max((len(r[0]) for r in rows), default=20)
    path_width = max(path_width, len("Folder"))

    header = f"{'Folder':<{path_width}}  {'Files':>6}  {'Size':>10}"
    print("\nDummy dataset summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    total_files = 0
    total_bytes = 0
    for path, count, size in rows:
        total_files += count
        total_bytes += size
        print(f"{path:<{path_width}}  {count:>6}  {_human_size(size):>10}")

    print("-" * len(header))
    print(
        f"{'TOTAL':<{path_width}}  {total_files:>6}  "
        f"{_human_size(total_bytes):>10}"
    )
    print(f"\nRoot: {dummy_root}")


def create_dummy_dataset(root: Path, seed: int = 42) -> Path:
    """Generate the full dummy dataset under ``root/data/dummy``."""
    rng = np.random.default_rng(seed)
    dummy_root = root / "data" / "dummy"

    n_images = generate_images(dummy_root / "images", rng)
    n_videos = generate_videos(dummy_root / "videos", rng)

    print(f"Generated {n_images} images and {n_videos} videos.")
    print_summary(dummy_root)
    return dummy_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a dummy SEETHRU dataset.")
    parser.add_argument(
        "--root",
        default=str(_repo_root()),
        help="Repository root under which data/dummy is created.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_dummy_dataset(Path(args.root), seed=args.seed)


if __name__ == "__main__":
    main()
