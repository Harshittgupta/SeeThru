"""Video processing for SEETHRU deepfake detection.

Turns FaceForensics++ and Celeb-DF v2 videos into fixed-length aligned face
sequences ready for a video classifier. Provides:

* :class:`DatasetConfig` — dataset paths and sampling config.
* :class:`FFPlusPlusLoader` — enumerate + identity-split FaceForensics++ videos.
* :class:`CelebDFLoader` — enumerate Celeb-DF v2 videos (test-only, used for
  cross-dataset generalization evaluation).
* :class:`VideoProcessor` — frame sampling, face extraction, dataset processing.
* :class:`DeepfakeVideoDataset` — a ``torch.utils.data.Dataset`` over processed
  sequences.

Run directly for a single-video sanity check::

    python ml/preprocessing/video_processor.py path/to/video.mp4
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Support running both as a module (``python -m ...``) and as a script.
try:
    from .face_detector import FaceDetector
except ImportError:  # pragma: no cover - direct-script execution
    from face_detector import FaceDetector

logger = logging.getLogger(__name__)

# FaceForensics++ default compression level and manipulation method folders.
FF_COMPRESSION = "c23"
FF_MANIPULATIONS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")

# Video file extensions to scan for.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# ImageNet normalization stats (shared with data.dataset_manager / augmentation).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REAL, FAKE = 0, 1


@dataclass
class DatasetConfig:
    """Paths and sampling configuration for the video datasets."""

    ff_root: str
    celebdf_root: str
    n_frames: int = 16
    face_size: int = 224


# --------------------------------------------------------------------------- #
# Identity-aware splitting helpers (shared by the loaders)
# --------------------------------------------------------------------------- #
def _split_by_identity(
    identities: List[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, set]:
    """Partition unique identities into train/val/test sets (no overlap)."""
    unique = sorted(set(identities))
    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(test_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)

    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test : n_test + n_val])
    train_ids = set(shuffled[n_test + n_val :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _balance_5050(videos: List[dict], seed: int) -> List[dict]:
    """Downsample the majority class so real:fake is 50:50."""
    reals = [v for v in videos if v["label"] == REAL]
    fakes = [v for v in videos if v["label"] == FAKE]
    k = min(len(reals), len(fakes))
    if k == 0:
        # One class missing — nothing to balance against; return as-is.
        return sorted(videos, key=lambda v: v["path"])

    rng = random.Random(seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    balanced = reals[:k] + fakes[:k]
    balanced.sort(key=lambda v: v["path"])
    return balanced


def _scan_videos(directory: Path) -> List[Path]:
    """Return sorted video files directly under ``directory`` (if it exists)."""
    if not directory.is_dir():
        logger.warning("Expected video directory not found: %s", directory)
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


# --------------------------------------------------------------------------- #
# FaceForensics++ loader
# --------------------------------------------------------------------------- #
class FFPlusPlusLoader:
    """Enumerate and identity-split FaceForensics++ videos.

    Expected layout (``c23`` compression)::

        <ff_root>/original_sequences/youtube/c23/videos/000.mp4 ...
        <ff_root>/manipulated_sequences/<Manip>/c23/videos/000_167.mp4 ...
    """

    def __init__(self, ff_root: str | Path, compression: str = FF_COMPRESSION) -> None:
        self.ff_root = Path(ff_root)
        self.compression = compression

    def _real_dir(self) -> Path:
        return (
            self.ff_root
            / "original_sequences"
            / "youtube"
            / self.compression
            / "videos"
        )

    def _fake_dir(self, manipulation: str) -> Path:
        return (
            self.ff_root
            / "manipulated_sequences"
            / manipulation
            / self.compression
            / "videos"
        )

    def get_video_paths(self, split: str = "train") -> List[dict]:
        """Scan all real and fake videos. (Splitting happens in get_split.)

        The ``split`` argument is accepted for API symmetry but does not filter
        here — :meth:`get_split` performs the identity-aware partition.
        """
        videos: List[dict] = []

        # Real videos: identity is the full stem, e.g. "000.mp4" -> "000".
        for path in _scan_videos(self._real_dir()):
            videos.append(
                {
                    "path": str(path),
                    "label": REAL,
                    "manipulation": None,
                    "identity": path.stem,
                }
            )

        # Fake videos: identity is the target (first) id, e.g. "000_167" -> "000".
        for manip in FF_MANIPULATIONS:
            for path in _scan_videos(self._fake_dir(manip)):
                identity = path.stem.split("_")[0]
                videos.append(
                    {
                        "path": str(path),
                        "label": FAKE,
                        "manipulation": manip,
                        "identity": identity,
                    }
                )

        logger.info("FF++ scan: found %d videos", len(videos))
        return videos

    def get_split(
        self,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> List[dict]:
        """Return the identity-separated, class-balanced videos for ``split``."""
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        videos = self.get_video_paths()
        id_splits = _split_by_identity(
            [v["identity"] for v in videos], val_ratio, test_ratio, seed
        )
        selected = id_splits[split]
        chosen = [v for v in videos if v["identity"] in selected]
        balanced = _balance_5050(chosen, seed)

        n_real = sum(v["label"] == REAL for v in balanced)
        n_fake = sum(v["label"] == FAKE for v in balanced)
        logger.info(
            "FF++ split '%s': %d videos (%d real / %d fake), %d identities",
            split,
            len(balanced),
            n_real,
            n_fake,
            len(selected),
        )
        return balanced


# --------------------------------------------------------------------------- #
# Celeb-DF v2 loader (test-only)
# --------------------------------------------------------------------------- #
class CelebDFLoader:
    """Enumerate Celeb-DF v2 videos for cross-dataset evaluation.

    Expected layout::

        <celebdf_root>/Celeb-real/id0_0000.mp4 ...
        <celebdf_root>/YouTube-real/00000.mp4 ...
        <celebdf_root>/Celeb-synthesis/id0_id1_0000.mp4 ...

    This loader is intentionally test-only: Celeb-DF is held out entirely for
    measuring generalization, never used for training.
    """

    REAL_DIRS = ("Celeb-real", "YouTube-real")
    FAKE_DIR = "Celeb-synthesis"

    def __init__(self, celebdf_root: str | Path) -> None:
        self.celebdf_root = Path(celebdf_root)

    @staticmethod
    def _identity(path: Path) -> str:
        # "id0_0000" -> "id0"; "id0_id1_0000" -> "id0"; "00000" -> "00000".
        return path.stem.split("_")[0]

    def get_video_paths(self) -> List[dict]:
        """Scan all real and fake Celeb-DF videos."""
        videos: List[dict] = []

        for real_dir in self.REAL_DIRS:
            for path in _scan_videos(self.celebdf_root / real_dir):
                videos.append(
                    {
                        "path": str(path),
                        "label": REAL,
                        "manipulation": None,
                        "identity": self._identity(path),
                    }
                )

        for path in _scan_videos(self.celebdf_root / self.FAKE_DIR):
            videos.append(
                {
                    "path": str(path),
                    "label": FAKE,
                    "manipulation": "Celeb-synthesis",
                    "identity": self._identity(path),
                }
            )

        n_real = sum(v["label"] == REAL for v in videos)
        logger.info(
            "Celeb-DF scan: %d videos (%d real / %d fake)",
            len(videos),
            n_real,
            len(videos) - n_real,
        )
        return videos


# --------------------------------------------------------------------------- #
# Video processing
# --------------------------------------------------------------------------- #
class VideoProcessor:
    """Sample frames, extract aligned faces, and process whole datasets."""

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        n_frames: int = 16,
        face_size: int = 224,
        max_missing: Optional[int] = None,
    ) -> None:
        self.n_frames = n_frames
        self.face_size = face_size
        self.face_detector = face_detector or FaceDetector(output_size=face_size)
        # Reject a video if more than this many frames have no detectable face.
        self.max_missing = max_missing if max_missing is not None else n_frames // 2

    # ------------------------------------------------------------------ #
    def extract_frames(
        self, video_path: str, n_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """Uniformly sample ``n_frames`` BGR frames across the full duration.

        If the video has fewer than ``n_frames`` frames, the last frame is
        repeated to pad the sequence to length.
        """
        n_frames = n_frames or self.n_frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                # Frame count unreliable for some codecs; fall back to reading.
                frames = self._read_all(cap)
            else:
                indices = np.linspace(0, total - 1, num=n_frames).round().astype(int)
                logger.debug(
                    "Sampling %s: total=%d, indices=%s",
                    video_path,
                    total,
                    indices.tolist(),
                )
                frames = self._read_indices(cap, indices)
        finally:
            cap.release()

        if not frames:
            raise IOError(f"No frames could be read from: {video_path}")

        # Pad by repeating the last frame if we came up short.
        if len(frames) < n_frames:
            logger.debug(
                "Padding %s: %d/%d frames, repeating last",
                video_path,
                len(frames),
                n_frames,
            )
            frames.extend([frames[-1].copy()] * (n_frames - len(frames)))

        return frames[:n_frames]

    @staticmethod
    def _read_indices(cap: cv2.VideoCapture, indices) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
            elif frames:
                # Seek/read failed near the end; reuse the last good frame.
                frames.append(frames[-1].copy())
        return frames

    @staticmethod
    def _read_all(cap: cv2.VideoCapture) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        return frames

    # ------------------------------------------------------------------ #
    def _build_face_sequence(self, video_path: str):
        """Core of :meth:`extract_face_sequence`; also returns stats.

        Returns ``(sequence_or_None, n_missing, n_total)``.
        """
        frames = self.extract_frames(video_path, self.n_frames)

        faces: List[Optional[np.ndarray]] = []
        missing_positions: List[int] = []
        for i, frame in enumerate(frames):
            detected = self.face_detector.detect_and_align(frame)
            if detected:
                # Single-subject videos: take the first detected face.
                faces.append(detected[0])
            else:
                faces.append(None)
                missing_positions.append(i)

        n_missing = len(missing_positions)
        n_total = len(frames)
        logger.debug(
            "%s: %d/%d frames without a face", video_path, n_missing, n_total
        )

        if n_missing > self.max_missing:
            logger.info(
                "Skipping %s: %d/%d frames have no face (> %d)",
                video_path,
                n_missing,
                n_total,
                self.max_missing,
            )
            return None, n_missing, n_total

        # Fill gaps by copying the nearest valid frame's face crop.
        if n_missing:
            self._interpolate_missing(faces)

        return faces, n_missing, n_total

    @staticmethod
    def _interpolate_missing(faces: List[Optional[np.ndarray]]) -> None:
        """Replace each ``None`` with the nearest valid face crop (in place)."""
        n = len(faces)
        for i in range(n):
            if faces[i] is not None:
                continue
            # Search outward for the nearest valid neighbour.
            for offset in range(1, n):
                lo, hi = i - offset, i + offset
                if lo >= 0 and faces[lo] is not None:
                    faces[i] = faces[lo].copy()
                    break
                if hi < n and faces[hi] is not None:
                    faces[i] = faces[hi].copy()
                    break

    def extract_face_sequence(self, video_path: str) -> Optional[List[np.ndarray]]:
        """Return ``n_frames`` aligned 224×224 face crops, or ``None``.

        Returns ``None`` when more than ``max_missing`` frames lack a detectable
        face (the video is considered unusable).
        """
        sequence, _missing, _total = self._build_face_sequence(video_path)
        return sequence

    # ------------------------------------------------------------------ #
    def process_dataset(self, loader, split: str = "train") -> List[dict]:
        """Process every video from ``loader`` for ``split`` into sequences.

        Accepts an :class:`FFPlusPlusLoader` (uses ``get_split``) or a
        :class:`CelebDFLoader` (test-only, uses ``get_video_paths``). Skips
        videos that come back unusable (``None``).
        """
        if hasattr(loader, "get_split"):
            videos = loader.get_split(split)
        else:
            videos = loader.get_video_paths()

        processed: List[dict] = []
        skipped = 0
        for n, video in enumerate(videos, 1):
            path = video["path"]
            try:
                sequence = self.extract_face_sequence(path)
            except (IOError, ValueError) as exc:
                logger.warning("Failed to process %s: %s", path, exc)
                sequence = None

            if sequence is None:
                skipped += 1
            else:
                processed.append(
                    {
                        "frames": sequence,
                        "label": video["label"],
                        "identity": video["identity"],
                        "manipulation": video.get("manipulation"),
                        "video_path": path,
                    }
                )
            if n % 50 == 0:
                logger.info("Processed %d/%d videos...", n, len(videos))

        logger.info(
            "Split '%s': %d processed, %d skipped (of %d)",
            split,
            len(processed),
            skipped,
            len(videos),
        )
        return processed


# --------------------------------------------------------------------------- #
# torch Dataset
# --------------------------------------------------------------------------- #
class DeepfakeVideoDataset(Dataset):
    """Dataset over processed face sequences for video deepfake detection.

    Each item is a 16-frame clip. The ``transform`` (an albumentations pipeline,
    wrapped to be torchvision-callable) is applied independently to each frame;
    frames are converted BGR→RGB first and normalized to ImageNet stats by the
    transform.
    """

    def __init__(self, processed: List[dict], transform=None) -> None:
        self.items = processed
        if transform is None:
            # Default: resize + ImageNet normalize (no augmentation).
            try:
                from .augmentation import build_val_transform
            except ImportError:  # pragma: no cover - direct-script execution
                from augmentation import build_val_transform
            transform = build_val_transform(image_size=224)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def _apply(self, rgb_frame: np.ndarray) -> torch.Tensor:
        """Apply the transform, supporting both wrapped and raw albumentations."""
        out = self.transform(rgb_frame)
        if isinstance(out, dict):  # raw albumentations Compose
            out = out["image"]
        return out

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        tensors = []
        for frame in item["frames"]:
            # Face crops are BGR (OpenCV); transforms expect RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(self._apply(rgb))

        frames = torch.stack(tensors, dim=0)  # (T, 3, H, W)
        return {
            "frames": frames,
            "label": item["label"],
            "identity": item["identity"],
        }


# --------------------------------------------------------------------------- #
# CLI sanity check
# --------------------------------------------------------------------------- #
def _sanity_check(video_path: str) -> None:
    processor = VideoProcessor()

    frames = processor.extract_frames(video_path)
    print(f"Frames extracted: {len(frames)}")

    sequence, n_missing, n_total = processor._build_face_sequence(video_path)
    success_rate = (n_total - n_missing) / n_total if n_total else 0.0
    print(
        f"Face detection success rate: {success_rate:.1%} "
        f"({n_total - n_missing}/{n_total} frames)"
    )

    if sequence is None:
        print("Output tensor shape: N/A (video unusable — too many missing faces)")
        return

    dataset = DeepfakeVideoDataset(
        [
            {
                "frames": sequence,
                "label": 0,
                "identity": "sanity",
                "manipulation": None,
                "video_path": video_path,
            }
        ]
    )
    sample = dataset[0]
    print(f"Output tensor shape: {tuple(sample['frames'].shape)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single-video sanity check.")
    parser.add_argument("video", help="Path to a video file.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _sanity_check(args.video)
