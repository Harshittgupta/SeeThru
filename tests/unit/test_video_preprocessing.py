"""Video preprocessing fixes (BUILD_PLAN T38/T40/T42).

The consequential test here is `test_clip_augmentation_is_consistent_across_frames`.
Per-frame augmentation on a video clip is not a small inefficiency -- it injects
frame-to-frame motion that dwarfs the subtle flicker the temporal branch exists
to detect, so the BiLSTM learns our augmentation noise instead of deepfake
artifacts. And the loss goes down the whole time, so nothing looks wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.preprocessing.video_processor import (
    MAX_DECODE_FRAMES,
    DeepfakeVideoDataset,
    FrameSample,
    VideoProcessor,
)


class FakeDetector:
    """A detector that always finds a face. Keeps TensorFlow out of the tests."""

    def __init__(self, output_size: int = 224, fail_on: set[int] | None = None) -> None:
        self.output_size = output_size
        self.fail_on = fail_on or set()
        self.calls = 0

    def detect_and_align(self, frame):
        i, self.calls = self.calls, self.calls + 1
        if i in self.fail_on:
            return []
        return [np.full((self.output_size, self.output_size, 3), i % 255, dtype=np.uint8)]


@pytest.fixture
def real_video(celebdf_root: Path) -> str:
    return str(next((celebdf_root / "Celeb-real").glob("*.mp4")))


# --------------------------------------------------------------------------- #
# T40: timing metadata
# --------------------------------------------------------------------------- #
def test_sample_frames_returns_timing_metadata(real_video: str):
    """fps + source_indices are what make frame->seconds possible.

    extract_frames used to compute the indices and throw them away, and never
    read CAP_PROP_FPS at all -- so the spec's manipulation timeline ("0-2s likely
    real, 2-5s suspicious") could not be built from its output.
    """
    proc = VideoProcessor(face_detector=FakeDetector(), n_frames=8)
    sample = proc.sample_frames(real_video, n_frames=8)

    assert len(sample.frames) == 8
    assert len(sample.source_indices) == 8
    assert sample.fps > 0
    assert sample.total_frames > 0
    assert sample.duration_s == pytest.approx(sample.total_frames / sample.fps)


def test_source_indices_are_ascending_and_in_range(real_video: str):
    proc = VideoProcessor(face_detector=FakeDetector(), n_frames=8)
    sample = proc.sample_frames(real_video, n_frames=8)

    assert sample.source_indices == sorted(sample.source_indices)
    assert all(0 <= i < sample.total_frames for i in sample.source_indices)


def test_timestamps_are_computable(real_video: str):
    """The whole point of T40."""
    proc = VideoProcessor(face_detector=FakeDetector(), n_frames=8)
    ts = proc.sample_frames(real_video, n_frames=8).timestamps()

    assert len(ts) == 8
    assert ts == sorted(ts)
    assert ts[0] >= 0.0


def test_sampling_spans_the_whole_video(real_video: str):
    """Uniform sampling must reach the end, not cluster at the start."""
    proc = VideoProcessor(face_detector=FakeDetector(), n_frames=8)
    sample = proc.sample_frames(real_video, n_frames=8)
    assert sample.source_indices[0] == 0
    assert sample.source_indices[-1] == sample.total_frames - 1


def test_extract_frames_still_works(real_video: str):
    """The old signature must keep working for callers that want frames only."""
    proc = VideoProcessor(face_detector=FakeDetector(), n_frames=4)
    frames = proc.extract_frames(real_video, n_frames=4)
    assert len(frames) == 4
    assert frames[0].ndim == 3


# --------------------------------------------------------------------------- #
# T42: sequential decode
# --------------------------------------------------------------------------- #
def test_sampled_indices_are_exact(real_video: str):
    """Sequential decode returns the frames we asked for.

    cap.set(CAP_PROP_POS_FRAMES) lands on the nearest KEYFRAME on H.264, so the
    old code sampled whatever keyframes happened to sit near the requested
    indices -- silently, and differently per video depending on its GOP layout.
    """
    proc = VideoProcessor(face_detector=FakeDetector(), n_frames=5)
    sample = proc.sample_frames(real_video, n_frames=5)

    expected = np.linspace(0, sample.total_frames - 1, num=5).round().astype(int)
    assert sample.source_indices == expected.tolist()


def test_decode_cap_constant_is_sane():
    """The cap on the unreliable-frame-count path is a security control (T55).

    That path is reachable whenever CAP_PROP_FRAME_COUNT <= 0, which a crafted or
    VFR file can arrange -- and it decodes into a Python list, ~6 MB/frame at
    1080p.
    """
    assert 1000 < MAX_DECODE_FRAMES <= 100_000


# --------------------------------------------------------------------------- #
# T38: per-clip augmentation -- the one that matters
# --------------------------------------------------------------------------- #
def _clip_item(n_frames: int = 8) -> dict:
    """A clip whose frames are IDENTICAL. Any output difference is augmentation."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return {
        "frames": [frame.copy() for _ in range(n_frames)],
        "label": 1,
        "identity": "id0",
        "manipulation": "Deepfakes",
        "video_path": "x.mp4",
    }


def test_clip_augmentation_is_consistent_across_frames():
    """ONE parameter draw per clip, not one per frame (T38).

    The frames going in are identical, so if the frames coming out differ, the
    transform was re-sampled per frame -- injecting flip/crop/rotation jitter
    into a sequence whose entire purpose is to expose subtle temporal
    inconsistency.
    """
    from ml.preprocessing.augmentation import build_train_transform

    ds = DeepfakeVideoDataset([_clip_item()], transform=build_train_transform())
    frames = ds[0]["frames"]

    first = frames[0]
    for i in range(1, len(frames)):
        assert torch.allclose(first, frames[i], atol=1e-6), (
            f"frame {i} differs from frame 0 despite identical input -- the "
            f"augmentation is being re-sampled per frame, so the temporal branch "
            f"would learn our own jitter instead of deepfake flicker"
        )


def test_clip_augmentation_is_consistent_on_DIFFERING_frames():
    """The strong version of the test above.

    The previous test feeds identical frames, so it cannot distinguish "the
    transform was replayed" from "same input, same output" -- it would pass even
    if ReplayCompose did nothing. Real clips have differing frames, and
    albumentations explicitly warns that some ops "could work incorrectly in
    ReplayMode because their params depend on targets", so this needs checking
    rather than assuming.

    Method: frames that differ (random background) but share a bright marker in
    one corner. If HorizontalFlip is replayed, the marker is on the same side in
    all 16 frames. If it is re-drawn per frame, roughly half the clip is mirrored.

    Measured: 8/8 clips internally consistent, and both sides seen across clips.
    """
    from ml.preprocessing.augmentation import build_train_transform

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(16):
        f = rng.integers(0, 60, (224, 224, 3), dtype=np.uint8)  # differs per frame
        f[:40, :40] = 255                                       # shared marker
        frames.append(f)
    item = {
        "frames": frames, "label": 1, "identity": "id0",
        "manipulation": "Deepfakes", "video_path": "x.mp4",
    }
    ds = DeepfakeVideoDataset([item], transform=build_train_transform())

    def marker_side(frame: torch.Tensor) -> str:
        top = frame.mean(0)[:60]
        return "L" if top[:, :60].mean() > top[:, -60:].mean() else "R"

    seen_across_clips = set()
    for _ in range(8):
        out = ds[0]["frames"]
        sides = {marker_side(out[t]) for t in range(out.shape[0])}
        assert len(sides) == 1, (
            f"one clip contained BOTH marker sides {sides} -- HorizontalFlip is "
            f"being re-drawn per frame, so half the clip is mirrored relative to "
            f"the rest and the BiLSTM sees motion we invented"
        )
        seen_across_clips |= sides

    assert seen_across_clips == {"L", "R"}, (
        f"only ever saw {seen_across_clips} across 8 clips -- consistency was "
        f"achieved by disabling augmentation, which is not the fix"
    )


def test_val_transform_clip_is_deterministic():
    from ml.preprocessing.augmentation import build_val_transform

    ds = DeepfakeVideoDataset([_clip_item()], transform=build_val_transform())
    assert torch.allclose(ds[0]["frames"], ds[0]["frames"])


def test_clip_shape_and_metadata():
    from ml.preprocessing.augmentation import build_val_transform

    item = DeepfakeVideoDataset([_clip_item(8)], transform=build_val_transform())[0]
    assert item["frames"].shape == (8, 3, 224, 224)
    assert item["manipulation"] == "Deepfakes"  # T19


# --------------------------------------------------------------------------- #
# Face sequence provenance
# --------------------------------------------------------------------------- #
def test_interpolated_frames_are_flagged(real_video: str):
    """A copied face is not an observation, and the timeline must know (T50).

    _interpolate_missing fills a faceless frame with a neighbour's crop. Without
    this flag the UI plots that duplicate as a measured point on a manipulation
    timeline, which is simply false.
    """
    proc = VideoProcessor(
        face_detector=FakeDetector(fail_on={2, 5}), n_frames=8, max_missing=4
    )
    seq = proc.build_face_sequence(real_video)

    assert seq.usable
    assert seq.n_missing == 2
    assert seq.interpolated[2] and seq.interpolated[5]
    assert not seq.interpolated[0]
    assert seq.face_rate == pytest.approx(6 / 8)


def test_unusable_video_is_reported_not_crashed(real_video: str):
    proc = VideoProcessor(
        face_detector=FakeDetector(fail_on=set(range(8))), n_frames=8, max_missing=2
    )
    seq = proc.build_face_sequence(real_video)
    assert not seq.usable
    assert seq.faces is None
    assert seq.face_rate == 0.0


def test_build_face_sequence_shim_still_works(real_video: str):
    proc = VideoProcessor(face_detector=FakeDetector(), n_frames=4)
    sequence, n_missing, n_total = proc._build_face_sequence(real_video)
    assert sequence is not None
    assert n_missing == 0
    assert n_total == 4


def test_frame_sample_padding_is_marked():
    sample = FrameSample(
        frames=[np.zeros((2, 2, 3), np.uint8)] * 4,
        source_indices=[0, 1, 2, 2],
        fps=30.0, total_frames=3, duration_s=0.1, n_padded=1,
    )
    assert not sample.is_padded(2)
    assert sample.is_padded(3)
