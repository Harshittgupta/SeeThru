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
import json
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# Dataset-shaped fixtures (BUILD_PLAN T12)
#
# The layouts below mirror FaceForensics++ and Celeb-DF v2 *exactly* in the one
# respect that matters: filenames. That is what lets us test the identity-split
# logic -- and specifically the two-identity leak of T15 -- without waiting weeks
# for an EULA and downloading 24 GB.
#
# The trap being reproduced: FF++ names a fake `<target>_<source>.mp4`, so
# `033_097.mp4` contains BOTH identity 033 (the scene) and identity 097 (the face
# swapped in). Grouping on the leading token alone puts `033_097` in train while
# `097`'s own real video sits in test -- the model trains on a face it is then
# tested against. `_FF_PAIRS` below deliberately includes reciprocal pairs
# (000_001 AND 001_000) so a naive splitter provably leaks.
# --------------------------------------------------------------------------- #

FF_COMPRESSION = "c23"
FF_MANIPULATIONS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FF_N_ORIGINALS = 12

# (target, source) pairs. Reciprocals are intentional -- see above.
FF_PAIRS = (
    ("000", "001"),
    ("001", "000"),  # reciprocal of the above: the leak, if you group on token 0
    ("002", "003"),
    ("003", "002"),  # reciprocal
    ("004", "005"),
    ("006", "007"),
    ("008", "009"),
    ("010", "011"),
)

CELEBDF_N_IDENTITIES = 6
CELEBDF_REAL_PER_ID = 2
CELEBDF_YOUTUBE_REAL = 4
# (target_id, source_id) -- again with reciprocals.
CELEBDF_PAIRS = (
    (0, 1),
    (1, 0),
    (2, 3),
    (3, 2),
    (4, 5),
    (5, 4),
)


def _write_video(path: Path, rng: np.random.Generator, n_frames: int = VIDEO_FRAMES) -> None:
    """Write a short random-noise mp4 at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, VIDEO_FPS, (IMG_SIZE, IMG_SIZE))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {path}")
    try:
        for _ in range(n_frames):
            writer.write(_random_image(rng))
    finally:
        writer.release()


def generate_ffpp_like(root: Path, rng: np.random.Generator) -> int:
    """Write a miniature FaceForensics++ tree. Returns videos written.

    Layout (matches the real thing, which FFPlusPlusLoader walks)::

        original_sequences/youtube/c23/videos/000.mp4 .. 011.mp4
        manipulated_sequences/Deepfakes/c23/videos/000_001.mp4 ..
        manipulated_sequences/Face2Face/c23/videos/...
        manipulated_sequences/FaceSwap/c23/videos/...
        manipulated_sequences/NeuralTextures/c23/videos/...
        splits/{train,val,test}.json
    """
    written = 0

    originals = root / "original_sequences" / "youtube" / FF_COMPRESSION / "videos"
    for i in range(FF_N_ORIGINALS):
        _write_video(originals / f"{i:03d}.mp4", rng)
        written += 1

    for manip in FF_MANIPULATIONS:
        manip_dir = root / "manipulated_sequences" / manip / FF_COMPRESSION / "videos"
        for target, source in FF_PAIRS:
            _write_video(manip_dir / f"{target}_{source}.mp4", rng)
            written += 1

    # The official splits are lists of identity PAIRS, and both members of a pair
    # always land in the same split -- that is precisely what makes them leak-free
    # and why T15 uses them instead of a custom splitter. Mirror that structure.
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    official = {
        "train": [["000", "001"], ["002", "003"]],
        "val": [["004", "005"], ["006", "007"]],
        "test": [["008", "009"], ["010", "011"]],
    }
    for name, pairs in official.items():
        (splits_dir / f"{name}.json").write_text(json.dumps(pairs), encoding="utf-8")

    return written


def generate_celebdf_like(root: Path, rng: np.random.Generator) -> int:
    """Write a miniature Celeb-DF v2 tree. Returns videos written.

    Layout (matches the real thing, which CelebDFLoader walks)::

        Celeb-real/id0_0000.mp4 ..
        Celeb-synthesis/id0_id1_0000.mp4 ..
        YouTube-real/00000.mp4 ..
        List_of_testing_videos.txt
    """
    written = 0

    for identity in range(CELEBDF_N_IDENTITIES):
        for clip in range(CELEBDF_REAL_PER_ID):
            _write_video(root / "Celeb-real" / f"id{identity}_{clip:04d}.mp4", rng)
            written += 1

    for i in range(CELEBDF_YOUTUBE_REAL):
        _write_video(root / "YouTube-real" / f"{i:05d}.mp4", rng)
        written += 1

    synthesis_names = []
    for target, source in CELEBDF_PAIRS:
        name = f"id{target}_id{source}_0000.mp4"
        _write_video(root / "Celeb-synthesis" / name, rng)
        synthesis_names.append(name)
        written += 1

    # The official benchmark subset. Real lines are "1 <path>", fake "0 <path>"
    # (Celeb-DF labels real=1). Every published cross-dataset number is reported
    # on this list, so T16 filters to it.
    lines = [f"1 Celeb-real/id{i}_0000.mp4" for i in range(CELEBDF_N_IDENTITIES)]
    lines += [f"0 Celeb-synthesis/{n}" for n in synthesis_names]
    (root / "List_of_testing_videos.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    return written


def _dir_stats(directory: Path) -> tuple[int, int]:
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


def _tree_stats(directory: Path) -> tuple[int, int]:
    """(file_count, total_bytes) for files anywhere under ``directory``."""
    count = 0
    size = 0
    if directory.is_dir():
        for f in directory.rglob("*"):
            if f.is_file():
                count += 1
                size += f.stat().st_size
    return count, size


def print_summary(dummy_root: Path) -> None:
    """Print a table of folder paths, file counts and on-disk size.

    Reports every folder actually written, including the dataset-shaped trees --
    a summary that under-reports what it created is worse than no summary.
    """
    flat: list[Path] = [
        dummy_root / "images" / "real",
        dummy_root / "images" / "fake",
        dummy_root / "videos" / "real",
        dummy_root / "videos" / "fake",
    ]
    # These are deep trees, so they need a recursive count rather than _dir_stats.
    trees: list[Path] = [dummy_root / "ffpp_like", dummy_root / "celebdf_like"]

    rows = [(str(f), *_dir_stats(f)) for f in flat]
    rows += [(str(t), *_tree_stats(t)) for t in trees if t.is_dir()]
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


def create_dummy_dataset(
    root: Path, seed: int = 42, dataset_shaped: bool = True
) -> Path:
    """Generate the full dummy dataset under ``root/data/dummy``.

    Args:
        root: Repository root; output goes to ``root/data/dummy``.
        seed: RNG seed.
        dataset_shaped: Also emit the miniature FF++ / Celeb-DF trees. These are
            what make the identity-split logic testable offline; the flat
            ``images/`` + ``videos/`` layout alone exercises none of the real
            naming traps.
    """
    rng = np.random.default_rng(seed)
    dummy_root = root / "data" / "dummy"

    n_images = generate_images(dummy_root / "images", rng)
    n_videos = generate_videos(dummy_root / "videos", rng)
    print(f"Generated {n_images} images and {n_videos} videos.")

    if dataset_shaped:
        n_ff = generate_ffpp_like(dummy_root / "ffpp_like", rng)
        n_cdf = generate_celebdf_like(dummy_root / "celebdf_like", rng)
        print(
            f"Generated {n_ff} FF++-shaped and {n_cdf} Celeb-DF-shaped videos "
            f"(incl. reciprocal swap pairs -- the T15 leak trap)."
        )

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
