"""Prepare processed deepfake video datasets for SEETHRU.

Enumerates FaceForensics++ (train/val/test, identity-split + balanced) and
Celeb-DF v2 (cross-dataset test set), extracts aligned 16-frame face sequences
for every video, and pickles the results to ``data/processed/``.

Usage::

    python ml/preprocessing/prepare_datasets.py \
        --ff_root /path/to/FaceForensics++ \
        --celebdf_root /path/to/Celeb-DF-v2
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

# Support running both as a module and as a script.
try:
    from .video_processor import (
        CelebDFLoader,
        DatasetConfig,
        FFPlusPlusLoader,
        VideoProcessor,
    )
except ImportError:  # pragma: no cover - direct-script execution
    from video_processor import (
        CelebDFLoader,
        DatasetConfig,
        FFPlusPlusLoader,
        VideoProcessor,
    )

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Repository root: ml/preprocessing/<file> -> SEETHRU/."""
    return Path(__file__).resolve().parents[2]


def _save_pickle(data: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote %d sequences -> %s", len(data), out_path)


def _counts(videos: List[dict]) -> Tuple[int, int, int]:
    """(total, real, fake) for a list of video dicts."""
    total = len(videos)
    real = sum(v["label"] == 0 for v in videos)
    return total, real, total - real


def prepare(config: DatasetConfig, output_dir: Path) -> List[dict]:
    """Process all splits and write pickles. Returns per-split summary rows."""
    processor = VideoProcessor(
        n_frames=config.n_frames, face_size=config.face_size
    )
    ff_loader = FFPlusPlusLoader(config.ff_root)
    celebdf_loader = CelebDFLoader(config.celebdf_root)

    summary: List[dict] = []

    # FaceForensics++ train/val/test.
    for split in ("train", "val", "test"):
        videos = ff_loader.get_split(split)
        total, real, fake = _counts(videos)
        processed = processor.process_dataset(ff_loader, split)
        _save_pickle(processed, output_dir / f"ff_{split}.pkl")
        summary.append(
            {
                "split": f"ff_{split}",
                "total": total,
                "real": real,
                "fake": fake,
                "skipped": total - len(processed),
            }
        )

    # Celeb-DF v2 cross-dataset test set (test-only).
    cdf_videos = celebdf_loader.get_video_paths()
    total, real, fake = _counts(cdf_videos)
    cdf_processed = processor.process_dataset(celebdf_loader, "test")
    _save_pickle(cdf_processed, output_dir / "celebdf_test.pkl")
    summary.append(
        {
            "split": "celebdf_test",
            "total": total,
            "real": real,
            "fake": fake,
            "skipped": total - len(cdf_processed),
        }
    )

    return summary


def print_summary(summary: List[dict], output_dir: Path) -> None:
    header = (
        f"{'Split':<14}{'Total':>8}{'Real':>8}{'Fake':>8}{'Skipped':>9}"
    )
    print("\nDataset preparation summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['split']:<14}{row['total']:>8}{row['real']:>8}"
            f"{row['fake']:>8}{row['skipped']:>9}"
        )
    print("-" * len(header))
    print(f"\nProcessed pickles written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deepfake video datasets.")
    parser.add_argument("--ff_root", required=True, help="FaceForensics++ root.")
    parser.add_argument("--celebdf_root", required=True, help="Celeb-DF v2 root.")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--face_size", type=int, default=224)
    parser.add_argument(
        "--output_dir",
        default=str(_repo_root() / "data" / "processed"),
        help="Where to write the .pkl files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = DatasetConfig(
        ff_root=args.ff_root,
        celebdf_root=args.celebdf_root,
        n_frames=args.n_frames,
        face_size=args.face_size,
    )
    output_dir = Path(args.output_dir)

    summary = prepare(config, output_dir)
    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
