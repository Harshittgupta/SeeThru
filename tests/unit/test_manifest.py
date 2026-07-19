"""Manifest + processed-dataset tests (BUILD_PLAN T41/T41b/T43).

The two that earn their keep:

* `test_npy_paths_do_not_collide_across_manipulations` -- FF++ reuses stems across
  its four methods, so keying on the stem would have four videos overwrite one
  file. Silently. Losing 3/4 of the fakes.
* `test_resume_skips_already_processed` -- a 3-hour extraction that cannot resume
  is a 3-hour extraction you run twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.manifest import (
    MANIFEST_NAME,
    ManifestRow,
    append_row,
    done_videos,
    filter_rows,
    read_manifest,
    summarize,
    validate,
    write_manifest,
)
from data.processed_dataset import DeepfakeClipDataset, DeepfakeFrameDataset

REAL, FAKE = 0, 1


def _row(**over) -> ManifestRow:
    base = dict(
        video_path="raw/ffpp/Deepfakes/033_097.mp4",
        npy_path="ffpp/Deepfakes/033_097.npy",
        dataset="ffpp",
        split="train",
        label=FAKE,
        identity="033",
        identities=["033", "097"],
        manipulation="Deepfakes",
        n_frames=4,
        fps=30.0,
        duration_s=17.4,
        total_frames=522,
        source_indices=[0, 174, 348, 521],
        n_missing=0,
        face_rate=1.0,
        interpolated=[False] * 4,
    )
    base.update(over)
    return ManifestRow(**base)


@pytest.fixture
def processed_root(tmp_path: Path) -> Path:
    """A tiny processed dataset: .npy files + manifest."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        label = FAKE if i % 2 else REAL
        manip = "Deepfakes" if label == FAKE else "original"
        rel = f"ffpp/{manip}/vid{i:02d}.npy"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, rng.integers(0, 256, (4, 224, 224, 3), dtype=np.uint8))
        rows.append(
            _row(
                video_path=f"raw/vid{i:02d}.mp4",
                npy_path=rel,
                label=label,
                identity=f"id{i}",
                identities=[f"id{i}"],
                manipulation=manip if label == FAKE else "none",
                split="train" if i < 4 else "val",
            )
        )
    write_manifest(rows, tmp_path / MANIFEST_NAME)
    return tmp_path


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
def test_row_round_trips():
    row = _row()
    assert ManifestRow.from_json(row.to_json()) == row


def test_append_then_read(tmp_path: Path):
    path = tmp_path / MANIFEST_NAME
    append_row(_row(video_path="a.mp4"), path)
    append_row(_row(video_path="b.mp4"), path)
    assert [r.video_path for r in read_manifest(path)] == ["a.mp4", "b.mp4"]


def test_missing_manifest_reads_as_empty(tmp_path: Path):
    assert read_manifest(tmp_path / "nope.jsonl") == []


def test_torn_line_is_skipped_not_fatal(tmp_path: Path, caplog):
    """A crash mid-append leaves a partial last line. Re-processing one video is
    the correct response; refusing to read the manifest is not."""
    path = tmp_path / MANIFEST_NAME
    append_row(_row(video_path="good.mp4"), path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"video_path": "torn", "npy_pa')  # killed mid-write

    with caplog.at_level("WARNING"):
        rows = read_manifest(path)
    assert [r.video_path for r in rows] == ["good.mp4"]
    assert "unreadable" in caplog.text


def test_both_identities_survive_the_round_trip():
    """The T15 fix has to reach the manifest, or splitting cannot be audited."""
    row = ManifestRow.from_json(_row().to_json())
    assert row.identities == ["033", "097"]


# --------------------------------------------------------------------------- #
# Resume (T43)
# --------------------------------------------------------------------------- #
def test_done_videos_is_the_skip_set(tmp_path: Path):
    path = tmp_path / MANIFEST_NAME
    append_row(_row(video_path="a.mp4"), path)
    append_row(_row(video_path="b.mp4"), path)
    assert done_videos(path) == {"a.mp4", "b.mp4"}


def test_done_videos_empty_when_no_manifest(tmp_path: Path):
    assert done_videos(tmp_path / MANIFEST_NAME) == set()


# --------------------------------------------------------------------------- #
# The collision trap
# --------------------------------------------------------------------------- #
def test_npy_paths_do_not_collide_across_manipulations():
    """FF++ reuses stems across all four methods (T41).

    `033_097.mp4` exists under Deepfakes, Face2Face, FaceSwap AND
    NeuralTextures. Keying the output on the stem alone means four videos write
    one file -- 3/4 of the fakes silently lost, and four manifest rows pointing
    at a single array.
    """
    from ml.preprocessing.prepare_datasets import _npy_path_for

    paths = {
        str(
            _npy_path_for(
                {
                    "path": f"raw/manipulated_sequences/{m}/c23/videos/033_097.mp4",
                    "dataset": "ffpp",
                    "manipulation": m,
                }
            )
        )
        for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    }
    assert len(paths) == 4, f"stems collided across methods: {paths}"


def test_real_and_fake_with_the_same_stem_do_not_collide():
    from ml.preprocessing.prepare_datasets import _npy_path_for

    real = _npy_path_for({"path": "raw/000.mp4", "dataset": "ffpp", "manipulation": None})
    fake = _npy_path_for(
        {"path": "raw/000.mp4", "dataset": "ffpp", "manipulation": "Deepfakes"}
    )
    assert real != fake


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_accepts_a_good_manifest(processed_root: Path):
    assert validate(read_manifest(processed_root / MANIFEST_NAME), processed_root) == []


def test_validate_flags_a_missing_npy(processed_root: Path):
    rows = read_manifest(processed_root / MANIFEST_NAME)
    (processed_root / rows[0].npy_path).unlink()
    problems = validate(rows, processed_root)
    assert any("file missing" in p for p in problems)


def test_validate_flags_missing_identities(tmp_path: Path):
    problems = validate([_row(identities=[])], tmp_path)
    assert any("no identities" in p for p in problems)


def test_validate_flags_duplicate_npy_paths(tmp_path: Path):
    problems = validate([_row(video_path="a.mp4"), _row(video_path="b.mp4")], tmp_path)
    assert any("duplicate npy_path" in p for p in problems)


def test_summarize_reports_the_label_prior(processed_root: Path):
    text = summarize(read_manifest(processed_root / MANIFEST_NAME))
    assert "ffpp/train" in text and "real" in text


# --------------------------------------------------------------------------- #
# T41b: the two views over one manifest
# --------------------------------------------------------------------------- #
def test_frame_view_expands_clips_to_frames(processed_root: Path):
    from ml.preprocessing.augmentation import build_val_transform

    ds = DeepfakeFrameDataset(processed_root, split="train", transform=build_val_transform())
    assert len(ds) == 4 * 4  # 4 train videos x 4 frames

    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label in (REAL, FAKE)


def test_clip_view_keeps_clips_whole(processed_root: Path):
    from ml.preprocessing.augmentation import build_val_transform

    ds = DeepfakeClipDataset(processed_root, split="train", transform=build_val_transform())
    assert len(ds) == 4  # 4 train videos

    item = ds[0]
    assert item["frames"].shape == (4, 3, 224, 224)
    assert "manipulation" in item  # T19


def test_both_views_read_the_same_manifest(processed_root: Path):
    """The point of T41b: one extraction, two views."""
    from ml.preprocessing.augmentation import build_val_transform

    frames = DeepfakeFrameDataset(processed_root, split="train", transform=build_val_transform())
    clips = DeepfakeClipDataset(processed_root, split="train", transform=build_val_transform())
    assert len(frames) == sum(r.n_frames for r in clips.rows)
    assert {r.video_path for r in frames.rows} == {r.video_path for r in clips.rows}


def test_frame_metadata_needs_no_pixels(processed_root: Path):
    from ml.preprocessing.augmentation import build_val_transform

    ds = DeepfakeFrameDataset(processed_root, split="train", transform=build_val_transform())
    meta = ds.metadata(1)
    assert meta["frame_index"] == 1
    assert meta["t_seconds"] == pytest.approx(174 / 30.0)  # source index / fps
    assert meta["interpolated"] is False
    assert "identities" in meta


def test_frame_view_exposes_labels_and_video_ids(processed_root: Path):
    """labels() feeds class weights (T16); video_ids() feeds frame->video
    aggregation (T28)."""
    from ml.preprocessing.augmentation import build_val_transform

    ds = DeepfakeFrameDataset(processed_root, split="train", transform=build_val_transform())
    assert len(ds.labels()) == len(ds)
    assert len(set(ds.video_ids())) == 4


def test_empty_split_raises(processed_root: Path):
    """Same lesson as T11: an empty split must be loud, not len()==0."""
    from ml.preprocessing.augmentation import build_val_transform

    with pytest.raises(ValueError, match=r"No manifest rows"):
        DeepfakeFrameDataset(processed_root, split="test", transform=build_val_transform())


def test_missing_manifest_error_names_the_fix(tmp_path: Path):
    from ml.preprocessing.augmentation import build_val_transform

    with pytest.raises(FileNotFoundError, match=r"prepare_datasets.py"):
        DeepfakeFrameDataset(tmp_path, split="train", transform=build_val_transform())


def test_npy_is_memory_mapped(processed_root: Path):
    """A clip is 2.4 MB and the frame view wants one frame of it.

    mmap reads the ~150 KB it needs and lets the OS page cache be shared across
    DataLoader workers, instead of each worker materialising every clip.
    """
    from ml.preprocessing.augmentation import build_val_transform

    ds = DeepfakeFrameDataset(processed_root, split="train", transform=build_val_transform())
    arr = ds._load(ds.rows[0])
    assert isinstance(arr, np.memmap), "clips must be memory-mapped, not fully read"


def test_filter_rows(processed_root: Path):
    rows = read_manifest(processed_root / MANIFEST_NAME)
    assert len(filter_rows(rows, split="train")) == 4
    assert len(filter_rows(rows, split="val")) == 2
    assert len(filter_rows(rows, dataset="celebdf")) == 0
