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

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:  # import for type checkers only -- see _import_face_detector
    from .face_detector import FaceDetector

logger = logging.getLogger(__name__)


def _import_face_detector() -> type:
    """Import FaceDetector lazily (BUILD_PLAN T39).

    `face_detector` imports `retinaface`, which imports **all of TensorFlow**.
    Measured with `python -X importtime`:

        tensorflow                       11.0 s
        ml.preprocessing.video_processor 17.9 s   (total, was)

    That cost was previously paid at module import, by everything that touched
    this file -- including `FFPlusPlusLoader`/`CelebDFLoader`, which only list
    directories and never detect a face. It was also paid *per DataLoader
    worker* (Windows uses spawn, so every worker re-imports), and by the backend
    at startup, where TF would additionally sit on the GPU alongside PyTorch.

    Deferring it to VideoProcessor construction means only the code that
    actually detects faces pays. T45 removes TensorFlow altogether.
    """
    try:
        from .face_detector import FaceDetector
    except ImportError:  # pragma: no cover - direct-script execution
        from face_detector import FaceDetector
    return FaceDetector

# FaceForensics++ default compression level and manipulation method folders.
FF_COMPRESSION = "c23"
FF_MANIPULATIONS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")

# Video file extensions to scan for.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# ImageNet normalization stats (shared with data.dataset_manager / augmentation).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REAL, FAKE = 0, 1

# Hard cap on the fallback decode path (see VideoProcessor._read_all). Reachable
# whenever a file reports an unreliable frame count, which a crafted or VFR file
# can arrange -- so an unbounded decode is a direct OOM. ~10 minutes at 30fps.
MAX_DECODE_FRAMES = 18_000


@dataclass
class FrameSample:
    """Sampled frames plus the timing metadata needed to build a timeline (T40).

    ``extract_frames`` used to return a bare list, computing the sample indices
    and then discarding them, and never reading FPS at all. Both are required to
    convert a frame index into a timestamp -- without which the spec's
    "manipulation timeline" cannot exist. They were already being computed.
    """

    frames: list[np.ndarray]
    #: True source index in the original video for each sampled frame.
    source_indices: list[int]
    fps: float
    total_frames: int
    duration_s: float
    #: Trailing frames that are duplicates of the last real frame, not
    #: independent observations. The UI must not render these as evidence (T50).
    n_padded: int = 0

    def timestamps(self) -> list[float]:
        """Seconds for each sampled frame. Empty when fps is unavailable."""
        if self.fps <= 0:
            return []
        return [i / self.fps for i in self.source_indices]

    def is_padded(self, i: int) -> bool:
        return i >= len(self.frames) - self.n_padded


@dataclass
class FaceSequence:
    """Aligned face crops for one video, plus provenance for each one.

    ``interpolated`` is the field that matters downstream. A face crop can be in
    this sequence for three different reasons -- detected, copied from a
    neighbouring frame because detection failed, or duplicated as padding -- and
    only the first is an observation. Collapsing them into one list of arrays
    (as the old code did) means the timeline cannot tell evidence from filler,
    and will happily plot a copied frame as a measured point (T50).
    """

    faces: list[np.ndarray] | None
    sample: FrameSample
    n_missing: int
    interpolated: list[bool]
    usable: bool

    @property
    def face_rate(self) -> float:
        """Fraction of sampled frames where a face was actually detected."""
        n = len(self.sample.frames)
        return (n - self.n_missing) / n if n else 0.0


@dataclass
class DatasetConfig:
    """Paths and sampling configuration for the video datasets."""

    ff_root: str
    celebdf_root: str
    n_frames: int = 16
    face_size: int = 224


# --------------------------------------------------------------------------- #
# Identity extraction (BUILD_PLAN T15)
#
# THE TRAP: FaceForensics++ names a fake `<target>_<source>.mp4`. `033_097.mp4`
# is identity 033's scene with identity 097's FACE swapped in -- so the video
# contains BOTH people. Celeb-DF does the same with `id3_id5_0001.mp4`.
#
# The old code did `path.stem.split("_")[0]`, keeping only the target. That put
# 033_097 in train (keyed '033') while 097's own real video could sit in test:
# the model trains on 097's face and is then evaluated on it. Simulated on a
# realistic FF++ layout, 828 of 4000 fakes leaked this way.
#
# Worse, swaps CHAIN. If 000_001 and 001_002 both exist, then 000, 001 and 002
# are transitively bound and must all travel to the same split together. That is
# a connected-components problem, not a per-video one -- hence the union-find
# below. See tests/unit/test_dummy_dataset_shapes.py for the executable proof.
# --------------------------------------------------------------------------- #
def ff_identities(path: Path) -> tuple[str, ...]:
    """Every identity appearing in an FF++ video filename.

        000.mp4      -> ("000",)            a real video: one person
        033_097.mp4  -> ("033", "097")      a fake: target AND source
    """
    return tuple(path.stem.split("_"))


def celebdf_identities(path: Path) -> tuple[str, ...]:
    """Every identity appearing in a Celeb-DF v2 video filename.

        id0_0000.mp4      -> ("id0",)           Celeb-real: person + clip index
        id0_id1_0000.mp4  -> ("id0", "id1")     Celeb-synthesis: target AND source
        00000.mp4         -> ("youtube_00000",) YouTube-real: no person id at all

    The trailing clip index is not an identity, so it is dropped. YouTube-real
    clips get a synthetic per-file id: they have no subject label, so the safest
    assumption is that each is its own person.
    """
    parts = path.stem.split("_")
    ids = tuple(p for p in parts if p.startswith("id"))
    if ids:
        return ids
    # YouTube-real: bare numeric stem, no subject information available.
    return (f"youtube_{path.stem}",)


def group_identities(videos: list[dict]) -> dict[str, str]:
    """Union-find over identity co-occurrence -> ``{identity: group_key}``.

    Identities that appear together in ANY video are bound into one group, and
    binding is transitive (000_001 + 001_002 => {000, 001, 002} is one group).
    Splitting must then be done over *groups*, never raw identities, or a face
    present in two videos can straddle the train/test boundary.

    Group keys are the lexicographically smallest member, so they are stable
    across runs regardless of scan order.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            parent[hi] = lo  # deterministic: smaller key always wins

    for video in videos:
        ids = video["identities"]
        for identity in ids:
            find(identity)
        for identity in ids[1:]:
            union(ids[0], identity)

    return {identity: find(identity) for identity in parent}


# --------------------------------------------------------------------------- #
# Identity-aware splitting helpers (shared by the loaders)
# --------------------------------------------------------------------------- #
def _split_by_group(
    groups: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, set]:
    """Partition unique identity *groups* into train/val/test (no overlap).

    Takes groups, not identities: see :func:`group_identities` for why the
    distinction is the whole point.
    """
    unique = sorted(set(groups))
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


def _balance_5050(videos: list[dict], seed: int) -> list[dict]:
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


def _scan_videos(directory: Path) -> list[Path]:
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

    def _splits_dir(self) -> Path:
        return self.ff_root / "splits"

    def load_official_splits(self) -> dict[str, str] | None:
        """Parse FF++'s own ``splits/{train,val,test}.json`` -> ``{identity: split}``.

        These files are lists of identity *pairs* (720/140/140), and both members
        of every pair are guaranteed to sit in the same split. That is exactly
        the property our custom splitter has to work to reconstruct -- so when
        the official files are present, use them and delete the guesswork.

        Bonus, and not a small one: it makes our numbers directly comparable to
        every published FF++ baseline, which all report on these splits.

        Returns ``None`` if the files aren't present.
        """
        splits_dir = self._splits_dir()
        mapping: dict[str, str] = {}
        for split in ("train", "val", "test"):
            path = splits_dir / f"{split}.json"
            if not path.is_file():
                logger.warning(
                    "FF++ official split file missing: %s -- falling back to "
                    "grouped identity splitting",
                    path,
                )
                return None
            with open(path, encoding="utf-8") as fh:
                pairs = json.load(fh)
            for pair in pairs:
                for identity in pair:
                    prior = mapping.get(identity)
                    if prior is not None and prior != split:
                        raise ValueError(
                            f"FF++ official splits are inconsistent: identity "
                            f"{identity!r} appears in both {prior!r} and {split!r}"
                        )
                    mapping[identity] = split
        logger.info(
            "FF++ official splits loaded: %d identities across train/val/test",
            len(mapping),
        )
        return mapping

    def get_video_paths(self, split: str = "train") -> list[dict]:
        """Scan all real and fake videos. (Splitting happens in get_split.)

        The ``split`` argument is accepted for API symmetry but does not filter
        here — :meth:`get_split` performs the identity-aware partition.
        """
        videos: list[dict] = []

        # Real videos: "000.mp4" -> one identity, ("000",).
        for path in _scan_videos(self._real_dir()):
            videos.append(
                {
                    "path": str(path),
                    "label": REAL,
                    "manipulation": None,
                    "identities": ff_identities(path),
                }
            )

        # Fake videos: "000_167.mp4" -> BOTH identities, ("000", "167").
        # Keeping only the first was the leak (T15).
        for manip in FF_MANIPULATIONS:
            for path in _scan_videos(self._fake_dir(manip)):
                videos.append(
                    {
                        "path": str(path),
                        "label": FAKE,
                        "manipulation": manip,
                        "identities": ff_identities(path),
                    }
                )

        # `identity` is retained as the canonical group key so downstream code
        # and audits have one stable string to group on.
        identity_to_group = group_identities(videos)
        for video in videos:
            video["identity"] = identity_to_group[video["identities"][0]]

        logger.info(
            "FF++ scan: found %d videos, %d identities in %d groups",
            len(videos),
            len(identity_to_group),
            len(set(identity_to_group.values())),
        )
        return videos

    def get_split(
        self,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        balance: bool | None = None,
    ) -> list[dict]:
        """Return the identity-separated videos for ``split``.

        Uses FF++'s official splits when available, otherwise falls back to a
        pair-aware grouped split (and says so, loudly).

        Args:
            balance: Downsample to 50:50 real/fake. Defaults to ``True`` for
                train and ``False`` for val/test -- see :func:`_balance_5050`.
                Balancing an evaluation split throws away data, destroys the
                per-manipulation breakdown, and buys nothing: AUC is
                prevalence-insensitive (T16).
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if balance is None:
            balance = split == "train"

        videos = self.get_video_paths()

        official = self.load_official_splits()
        if official is not None:
            chosen = [
                v
                for v in videos
                if all(official.get(i) == split for i in v["identities"])
            ]
            self._warn_on_unmapped(videos, official)
            source = "official splits/*.json"
        else:
            groups = _split_by_group(
                [v["identity"] for v in videos], val_ratio, test_ratio, seed
            )
            selected = groups[split]
            chosen = [v for v in videos if v["identity"] in selected]
            source = f"grouped fallback (seed={seed})"

        result = _balance_5050(chosen, seed) if balance else sorted(
            chosen, key=lambda v: v["path"]
        )

        n_real = sum(v["label"] == REAL for v in result)
        n_fake = sum(v["label"] == FAKE for v in result)
        logger.info(
            "FF++ split '%s' via %s: %d videos (%d real / %d fake), "
            "%d groups, balance=%s",
            split,
            source,
            len(result),
            n_real,
            n_fake,
            len({v["identity"] for v in result}),
            balance,
        )
        return result

    @staticmethod
    def _warn_on_unmapped(videos: list[dict], official: dict[str, str]) -> None:
        """Flag videos the official splits don't cover, rather than dropping them silently.

        A video whose identities aren't all in one official split is excluded
        from every split -- which is correct, but silently losing data is how you
        end up wondering why your training set is smaller than the paper's.
        """
        unmapped = {
            i for v in videos for i in v["identities"] if i not in official
        }
        if unmapped:
            logger.warning(
                "%d identities are absent from FF++'s official splits and their "
                "videos will be excluded: %s",
                len(unmapped),
                sorted(unmapped)[:10],
            )


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
    TESTING_LIST = "List_of_testing_videos.txt"

    # Celeb-DF's own list labels real=1, fake=0 -- the OPPOSITE of this project's
    # REAL=0 / FAKE=1. Getting this backwards silently inverts every metric, so
    # it is spelled out rather than inlined.
    _CELEBDF_LABEL_TO_OURS = {"1": REAL, "0": FAKE}

    def __init__(self, celebdf_root: str | Path) -> None:
        self.celebdf_root = Path(celebdf_root)

    def load_testing_list(self) -> list[dict] | None:
        """Parse ``List_of_testing_videos.txt`` -> the official 518-video subset.

        Every published Celeb-DF cross-dataset number is reported on this subset
        (178 real / 340 fake). Evaluating on all 6,529 videos instead makes our
        headline metric incomparable to the literature -- and since that set is
        86.4% fake, an always-"fake" model would score 86.4% on it (T16).

        Returns ``None`` if the file isn't present.
        """
        path = self.celebdf_root / self.TESTING_LIST
        if not path.is_file():
            logger.warning(
                "Celeb-DF official testing list missing: %s -- falling back to "
                "the FULL video set, which is NOT comparable to published "
                "baselines and is heavily fake-skewed",
                path,
            )
            return None

        videos: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    label_str, rel_path = line.split(" ", 1)
                    label = self._CELEBDF_LABEL_TO_OURS[label_str]
                except (ValueError, KeyError) as exc:
                    raise ValueError(
                        f"{path}:{lineno}: cannot parse {line!r} "
                        f"(expected '<0|1> <relative/path.mp4>')"
                    ) from exc

                video_path = self.celebdf_root / rel_path
                videos.append(
                    {
                        "path": str(video_path),
                        "label": label,
                        "manipulation": "Celeb-synthesis" if label == FAKE else None,
                        "identities": celebdf_identities(video_path),
                    }
                )

        missing = [v["path"] for v in videos if not Path(v["path"]).is_file()]
        if missing:
            logger.warning(
                "%d/%d videos in the official testing list are missing on disk "
                "(first: %s)",
                len(missing),
                len(videos),
                missing[0],
            )

        n_real = sum(v["label"] == REAL for v in videos)
        logger.info(
            "Celeb-DF official testing list: %d videos (%d real / %d fake)",
            len(videos),
            n_real,
            len(videos) - n_real,
        )
        return videos

    def get_video_paths(self, official_only: bool = True) -> list[dict]:
        """Scan Celeb-DF videos.

        Args:
            official_only: Use ``List_of_testing_videos.txt`` (the 518-video
                published benchmark) when available. Defaults to ``True``: the
                full set is 86.4% fake and incomparable to any baseline. Pass
                ``False`` only when you deliberately want the whole corpus.
        """
        if official_only:
            official = self.load_testing_list()
            if official is not None:
                self._assign_groups(official)
                return official

        videos: list[dict] = []

        for real_dir in self.REAL_DIRS:
            for path in _scan_videos(self.celebdf_root / real_dir):
                videos.append(
                    {
                        "path": str(path),
                        "label": REAL,
                        "manipulation": None,
                        "identities": celebdf_identities(path),
                    }
                )

        # "id0_id1_0000.mp4" -> BOTH identities. Keeping only the first was the
        # same leak as FF++ (T15).
        for path in _scan_videos(self.celebdf_root / self.FAKE_DIR):
            videos.append(
                {
                    "path": str(path),
                    "label": FAKE,
                    "manipulation": "Celeb-synthesis",
                    "identities": celebdf_identities(path),
                }
            )

        self._assign_groups(videos)

        n_real = sum(v["label"] == REAL for v in videos)
        logger.info(
            "Celeb-DF scan (full set): %d videos (%d real / %d fake)",
            len(videos),
            n_real,
            len(videos) - n_real,
        )
        return videos

    @staticmethod
    def _assign_groups(videos: list[dict]) -> None:
        """Attach the canonical identity-group key to each video, in place."""
        identity_to_group = group_identities(videos)
        for video in videos:
            video["identity"] = identity_to_group[video["identities"][0]]


# --------------------------------------------------------------------------- #
# Video processing
# --------------------------------------------------------------------------- #
class VideoProcessor:
    """Sample frames, extract aligned faces, and process whole datasets."""

    def __init__(
        self,
        face_detector: FaceDetector | None = None,
        n_frames: int = 16,
        face_size: int = 224,
        max_missing: int | None = None,
    ) -> None:
        self.n_frames = n_frames
        self.face_size = face_size
        # Lazy: constructing a default detector is what drags in TensorFlow.
        # Passing one in (e.g. a fake, in tests) skips the import entirely.
        if face_detector is None:
            face_detector = _import_face_detector()(output_size=face_size)
        self.face_detector = face_detector
        # Reject a video if more than this many frames have no detectable face.
        self.max_missing = max_missing if max_missing is not None else n_frames // 2

    # ------------------------------------------------------------------ #
    def sample_frames(
        self, video_path: str, n_frames: int | None = None
    ) -> FrameSample:
        """Uniformly sample ``n_frames`` BGR frames, **keeping the timing metadata**.

        Returns a :class:`FrameSample` carrying fps, the true source index of
        every sampled frame, the duration, and which frames are padding.

        The old ``extract_frames`` computed ``indices = np.linspace(...)`` and then
        **threw them away**, and never read ``CAP_PROP_FPS`` at all (T40). Without
        both, frame -> seconds is not computable, which makes the spec's entire
        "manipulation timeline" (0-2s likely real, 2-5s suspicious) impossible to
        build. Everything needed was already being computed and discarded.
        """
        n_frames = n_frames or self.n_frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

            if total <= 0:
                # Some codecs report an unreliable frame count. Decode, but with a
                # hard cap: an unbounded read of a crafted/VFR file is a direct
                # OOM (the backend gates this with ffprobe too -- T55).
                frames, indices = self._read_all(cap, limit=MAX_DECODE_FRAMES)
                total = len(frames)
                if frames:
                    keep = np.linspace(0, len(frames) - 1, num=n_frames).round().astype(int)
                    indices = [indices[i] for i in keep]
                    frames = [frames[i] for i in keep]
            else:
                wanted = np.linspace(0, total - 1, num=n_frames).round().astype(int)
                frames, indices = self._read_sequential(cap, wanted)
        finally:
            cap.release()

        if not frames:
            raise OSError(f"No frames could be read from: {video_path}")

        n_real = len(frames)
        # Pad by repeating the last frame if we came up short. Padded frames are
        # duplicates that were never independently observed, so they are marked:
        # the UI must not draw them as measured evidence on a timeline (T50).
        if n_real < n_frames:
            logger.debug(
                "Padding %s: %d/%d frames, repeating last",
                video_path, n_real, n_frames,
            )
            frames.extend([frames[-1].copy()] * (n_frames - n_real))
            indices.extend([indices[-1]] * (n_frames - n_real))

        if fps <= 0:
            logger.warning(
                "%s reports fps=%.2f; timestamps will be unavailable for this "
                "video and the timeline will fall back to frame indices.",
                video_path, fps,
            )

        return FrameSample(
            frames=frames[:n_frames],
            source_indices=[int(i) for i in indices[:n_frames]],
            fps=fps,
            total_frames=int(total),
            duration_s=(total / fps) if fps > 0 else 0.0,
            n_padded=max(0, n_frames - n_real),
        )

    def extract_frames(
        self, video_path: str, n_frames: int | None = None
    ) -> list[np.ndarray]:
        """Frames only. Thin wrapper over :meth:`sample_frames` for callers that
        genuinely do not need timing."""
        return self.sample_frames(video_path, n_frames).frames

    @staticmethod
    def _read_sequential(
        cap: cv2.VideoCapture, wanted
    ) -> tuple[list[np.ndarray], list[int]]:
        """Decode forward, keeping only the wanted indices (T42).

        Replaces a ``cap.set(CAP_PROP_POS_FRAMES, i)`` seek per frame, which was
        both slower and **wrong**:

        * **Wrong**: on inter-frame codecs (FF++ is H.264 c23) a seek lands on the
          nearest *keyframe*, not the frame you asked for. So the "uniform
          sampling" was actually sampling whatever keyframes happened to be near
          the requested indices -- silently, and differently per video depending
          on its GOP structure.
        * **Slow**: each seek forces a re-decode from the preceding keyframe,
          ~50-200 ms. For FF++ that is 5000 videos x 16 seeks = 80k seeks, hours
          of pure seeking.

        Decoding forward costs one pass to the last wanted index -- for a ~500
        frame FF++ clip, ~500 cheap decodes beats 16 expensive seeks, and every
        returned frame is exactly the one requested.

        (For very long videos the arithmetic would flip back toward seeking, but
        FF++/Celeb-DF clips are ~300-500 frames, so forward decode wins.)
        """
        want = sorted({int(i) for i in wanted})
        want_set = set(want)
        last = want[-1]

        frames: list[np.ndarray] = []
        got: list[int] = []
        idx = 0
        while idx <= last:
            ok, frame = cap.read()
            if not ok or frame is None:
                break  # short/truncated video; caller pads
            if idx in want_set:
                frames.append(frame)
                got.append(idx)
            idx += 1
        return frames, got

    @staticmethod
    def _read_all(
        cap: cv2.VideoCapture, limit: int = MAX_DECODE_FRAMES
    ) -> tuple[list[np.ndarray], list[int]]:
        """Decode up to ``limit`` frames. Used only when the frame count is unreliable.

        **The limit is a security control, not tidiness.** This path runs whenever
        ``CAP_PROP_FRAME_COUNT <= 0``, which a crafted or variable-frame-rate file
        can trigger at will -- and without a cap it decodes the entire video into
        a Python list. At 1080p that is ~6 MB per frame, so a long file is a
        direct OOM. The backend gates uploads with ffprobe as well (T55), but this
        function is reachable from the offline pipeline too and must not rely on
        a caller it cannot see.
        """
        frames: list[np.ndarray] = []
        idx = 0
        while idx < limit:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            idx += 1
        if idx >= limit:
            logger.warning(
                "Hit the %d-frame decode cap on a video with an unreliable frame "
                "count; truncating. Sampling will cover only the first %d frames.",
                limit, limit,
            )
        return frames, list(range(len(frames)))

    # ------------------------------------------------------------------ #
    def build_face_sequence(self, video_path: str) -> FaceSequence:
        """Extract aligned face crops plus everything downstream needs (T40).

        Public (was ``_build_face_sequence``): the backend needs the per-video
        face-detection stats to answer ``insufficient_faces`` honestly, and
        reaching into a private method for them is worse than exporting one.
        """
        sample = self.sample_frames(video_path, self.n_frames)

        faces: list[np.ndarray | None] = []
        missing_positions: list[int] = []
        for i, frame in enumerate(sample.frames):
            detected = self.face_detector.detect_and_align(frame)
            if detected:
                # Single-subject videos: take the first detected face.
                faces.append(detected[0])
            else:
                faces.append(None)
                missing_positions.append(i)

        n_missing = len(missing_positions)
        n_total = len(sample.frames)
        logger.debug("%s: %d/%d frames without a face", video_path, n_missing, n_total)

        if n_missing > self.max_missing:
            logger.info(
                "Skipping %s: %d/%d frames have no face (> %d)",
                video_path, n_missing, n_total, self.max_missing,
            )
            return FaceSequence(
                faces=None, sample=sample, n_missing=n_missing,
                interpolated=[], usable=False,
            )

        # Which frames carry a face we actually observed, vs one copied from a
        # neighbour below. This distinction has to survive to the UI: an
        # interpolated frame is a duplicate, not evidence, and drawing it on a
        # manipulation timeline as a measured point is a straightforward lie (T50).
        interpolated = [f is None for f in faces]
        # Padded frames are duplicates too, for a different reason.
        for i in range(len(faces) - sample.n_padded, len(faces)):
            if 0 <= i < len(interpolated):
                interpolated[i] = True

        if n_missing:
            self._interpolate_missing(faces)

        return FaceSequence(
            faces=faces, sample=sample, n_missing=n_missing,
            interpolated=interpolated, usable=True,
        )

    def _build_face_sequence(self, video_path: str):
        """Back-compat shim → ``(sequence_or_None, n_missing, n_total)``."""
        seq = self.build_face_sequence(video_path)
        return (
            seq.faces if seq.usable else None,
            seq.n_missing,
            len(seq.sample.frames),
        )

    @staticmethod
    def _interpolate_missing(faces: list[np.ndarray | None]) -> None:
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

    def extract_face_sequence(self, video_path: str) -> list[np.ndarray] | None:
        """Return ``n_frames`` aligned 224×224 face crops, or ``None``.

        Returns ``None`` when more than ``max_missing`` frames lack a detectable
        face (the video is considered unusable).
        """
        sequence, _missing, _total = self._build_face_sequence(video_path)
        return sequence

    # ------------------------------------------------------------------ #
    def process_dataset(self, loader, split: str = "train") -> list[dict]:
        """Process every video from ``loader`` for ``split`` into sequences.

        Accepts an :class:`FFPlusPlusLoader` (uses ``get_split``) or a
        :class:`CelebDFLoader` (test-only, uses ``get_video_paths``). Skips
        videos that come back unusable (``None``).
        """
        if hasattr(loader, "get_split"):
            videos = loader.get_split(split)
        else:
            videos = loader.get_video_paths()

        processed: list[dict] = []
        failures: list[dict] = []
        for n, video in enumerate(videos, 1):
            path = video["path"]
            try:
                seq = self.build_face_sequence(path)
            except (OSError, ValueError) as exc:
                logger.warning("Failed to process %s: %s", path, exc)
                failures.append(
                    {"path": path, "label": video["label"], "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue

            if not seq.usable:
                failures.append(
                    {
                        "path": path,
                        "label": video["label"],
                        "reason": f"insufficient_faces ({seq.n_missing}/"
                                  f"{len(seq.sample.frames)} frames had none)",
                    }
                )
            else:
                processed.append(
                    {
                        "frames": seq.faces,
                        "label": video["label"],
                        "identity": video["identity"],
                        "identities": video.get("identities", ()),
                        "manipulation": video.get("manipulation"),
                        "video_path": path,
                        # Timing + provenance (T40). Without these, frame->seconds
                        # is not computable and the timeline cannot distinguish an
                        # observed frame from a copied one.
                        "fps": seq.sample.fps,
                        "source_indices": seq.sample.source_indices,
                        "duration_s": seq.sample.duration_s,
                        "total_frames": seq.sample.total_frames,
                        "interpolated": seq.interpolated,
                        "face_rate": seq.face_rate,
                    }
                )
            if n % 50 == 0:
                logger.info("Processed %d/%d videos...", n, len(videos))

        # Report WHICH videos were lost and why, not just how many. A bare count
        # hides the question that matters: were the failures disproportionately
        # real or fake? That is a silent shift in the label prior, and it changes
        # every metric downstream (T43).
        if failures:
            n_real_lost = sum(f["label"] == REAL for f in failures)
            logger.warning(
                "Split '%s': %d videos unusable (%d real / %d fake). First: %s",
                split, len(failures), n_real_lost, len(failures) - n_real_lost,
                failures[0]["path"],
            )
        logger.info(
            "Split '%s': %d processed, %d skipped (of %d)",
            split, len(processed), len(failures), len(videos),
        )
        self.last_failures = failures
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

    def __init__(self, processed: list[dict], transform=None) -> None:
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

    def _apply_clip(self, rgb_frames: list[np.ndarray]) -> list[torch.Tensor]:
        """Apply ONE sampled transform to every frame in the clip (T38).

        Uses ``A.ReplayCompose`` when the underlying pipeline supports it:
        transform the first frame, capture the parameters that were actually
        drawn, then replay exactly those on the remaining 15. The clip is
        augmented as a unit, so the only motion in it is the motion that was
        filmed.

        Falls back to per-frame application for a deterministic transform (e.g.
        ``val_transform``), where there is nothing random to keep consistent.
        """
        replay = self._replay_pipeline()
        if replay is None:
            return [self._apply(f) for f in rgb_frames]

        first = replay(image=rgb_frames[0])
        params = first["replay"]
        out = [first["image"]]
        out.extend(
            A.ReplayCompose.replay(params, image=f)["image"] for f in rgb_frames[1:]
        )
        return out

    def _replay_pipeline(self):
        """The wrapped albumentations pipeline as a ReplayCompose, or None.

        Cached: rebuilding it per __getitem__ would be pure overhead in the
        DataLoader hot path.
        """
        if hasattr(self, "_replay_cache"):
            return self._replay_cache

        pipeline = getattr(self.transform, "transform", None)  # AlbumentationsTransform
        if pipeline is None and isinstance(self.transform, A.BaseCompose):
            pipeline = self.transform

        replay = None
        if isinstance(pipeline, A.BaseCompose):
            # Rebuild the same op list as a ReplayCompose. Reusing the ops
            # themselves keeps this in lockstep with augmentation.py -- copying
            # the list here would drift the moment someone edits _train_aug_list.
            replay = A.ReplayCompose(list(pipeline.transforms))
        self._replay_cache = replay
        return replay

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]

        # ONE parameter draw for the whole clip (T38).
        #
        # This previously called the transform independently per frame, so each
        # of the 16 frames drew its own RandomCrop offset, its own rotation
        # angle, and its own coin-flip for HorizontalFlip -- meaning roughly half
        # a clip came out mirrored relative to the other half, and the crop
        # jittered frame to frame.
        #
        # For an image model that is merely wasteful. For the temporal branch it
        # is fatal: a BiLSTM over that sequence sees enormous frame-to-frame
        # motion that we injected ourselves, dwarfing the subtle flicker and
        # identity instability it exists to detect. It would learn our
        # augmentation noise instead of deepfake artifacts, and the loss would
        # fall the whole time.
        #
        # ReplayCompose records the parameters of the first call and replays the
        # identical transform on the rest of the clip.
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in item["frames"]]
        tensors = self._apply_clip(rgb_frames)

        frames = torch.stack(tensors, dim=0)  # (T, 3, H, W)
        return {
            "frames": frames,
            "label": item["label"],
            "identity": item["identity"],
            # `manipulation` was collected all the way through process_dataset and
            # then dropped here (T19). Without it the per-method breakdown is
            # impossible -- and that breakdown is where the real finding lives:
            # Deepfakes/Face2Face/FaceSwap all score ~0.98-0.99 while
            # NeuralTextures sits at ~0.90-0.95, because it only edits the mouth.
            # A single averaged AUC hides that completely.
            #
            # None for real videos. The default collate turns a list of None into
            # a list, which is fine; keep it a string for fakes.
            "manipulation": item.get("manipulation") or "none",
            "video_path": item.get("video_path", ""),
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
