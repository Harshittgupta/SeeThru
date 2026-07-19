"""FF++ / Celeb-DF loader tests -- the two-identity leak (BUILD_PLAN T15, T16).

The bug these close: FF++ names a fake `<target>_<source>.mp4`, so `033_097.mp4`
contains BOTH faces. The old code kept only the target (`stem.split("_")[0]`),
so `033_097` could sit in train while `097`'s own real video sat in test -- the
model training on a face it is then evaluated against.

`test_ffpp_naive_leading_token_split_provably_leaks` in test_dummy_dataset_shapes.py
demonstrates the bug. These tests assert the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.preprocessing.video_processor import (
    CelebDFLoader,
    FFPlusPlusLoader,
    celebdf_identities,
    ff_identities,
    group_identities,
)

REAL, FAKE = 0, 1


# --------------------------------------------------------------------------- #
# Identity extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("000", ("000",)),  # real: one person
        ("033_097", ("033", "097")),  # fake: target AND source -- the whole point
        ("001_000", ("001", "000")),  # the reciprocal
    ],
)
def test_ff_identities_keeps_both(stem: str, expected: tuple):
    assert ff_identities(Path(f"{stem}.mp4")) == expected


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("id0_0000", ("id0",)),  # Celeb-real: clip index is not an identity
        ("id3_id5_0001", ("id3", "id5")),  # Celeb-synthesis: both
        ("00000", ("youtube_00000",)),  # YouTube-real: no subject label at all
    ],
)
def test_celebdf_identities_keeps_both(stem: str, expected: tuple):
    assert celebdf_identities(Path(f"{stem}.mp4")) == expected


# --------------------------------------------------------------------------- #
# Union-find grouping
# --------------------------------------------------------------------------- #
def test_group_identities_binds_a_swap_pair():
    videos = [{"identities": ("000", "001")}]
    groups = group_identities(videos)
    assert groups["000"] == groups["001"], "a swap pair must share a group"


def test_group_identities_is_transitive():
    """Swap chains bind transitively.

    000_001 and 001_002 means 000, 001 and 002 must ALL travel together --
    a per-video rule would miss this and let 000 and 002 be separated.
    """
    videos = [{"identities": ("000", "001")}, {"identities": ("001", "002")}]
    groups = group_identities(videos)
    assert groups["000"] == groups["001"] == groups["002"]


def test_group_identities_keeps_strangers_apart():
    """Grouping must not over-merge, or everything collapses into one group."""
    videos = [{"identities": ("000", "001")}, {"identities": ("008", "009")}]
    groups = group_identities(videos)
    assert groups["000"] != groups["008"]


def test_group_identities_is_deterministic():
    """Group keys must not depend on scan order."""
    a = group_identities([{"identities": ("000", "001")}, {"identities": ("001", "002")}])
    b = group_identities([{"identities": ("001", "002")}, {"identities": ("000", "001")}])
    assert a == b


# --------------------------------------------------------------------------- #
# FF++ loader
# --------------------------------------------------------------------------- #
def test_ffpp_scan_records_both_identities(ffpp_root: Path):
    loader = FFPlusPlusLoader(ffpp_root)
    videos = loader.get_video_paths()
    assert videos

    fakes = [v for v in videos if v["label"] == FAKE]
    assert fakes
    for v in fakes:
        assert len(v["identities"]) == 2, f"fake {v['path']} lost an identity"


def test_ffpp_official_splits_are_used(ffpp_root: Path):
    loader = FFPlusPlusLoader(ffpp_root)
    mapping = loader.load_official_splits()
    assert mapping is not None
    assert mapping["000"] == mapping["001"] == "train"
    assert mapping["008"] == mapping["009"] == "test"


def test_ffpp_official_splits_absent_falls_back(ffpp_root: Path, tmp_path: Path):
    """No splits/*.json -> fall back rather than crash, but warn."""
    import shutil

    clone = tmp_path / "no_splits"
    shutil.copytree(ffpp_root, clone)
    shutil.rmtree(clone / "splits")

    loader = FFPlusPlusLoader(clone)
    assert loader.load_official_splits() is None
    assert loader.get_split("train")  # still works via the grouped fallback


def test_ffpp_no_identity_leak_across_splits(ffpp_root: Path):
    """THE test. No identity may appear in two splits -- counting BOTH ids of
    every fake, which is exactly what the old code failed to do."""
    loader = FFPlusPlusLoader(ffpp_root)
    per_split = {
        name: loader.get_split(name, balance=False) for name in ("train", "val", "test")
    }

    ids = {
        name: {i for v in videos for i in v["identities"]}
        for name, videos in per_split.items()
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a] & ids[b]
        assert not overlap, f"identity leak between {a} and {b}: {sorted(overlap)}"


def test_ffpp_reciprocal_pairs_land_together(ffpp_root: Path):
    """000_001 and 001_000 contain the same two faces, so they must never be
    separated. This is the precise case the old splitter got wrong."""
    loader = FFPlusPlusLoader(ffpp_root)
    where = {}
    for name in ("train", "val", "test"):
        for v in loader.get_split(name, balance=False):
            where[Path(v["path"]).stem] = name

    assert where.get("000_001") == where.get("001_000") != None  # noqa: E711
    assert where.get("002_003") == where.get("003_002") != None  # noqa: E711


def test_ffpp_splits_are_non_empty(ffpp_root: Path):
    """Same lesson as T13: a leak check over empty splits is a false green."""
    loader = FFPlusPlusLoader(ffpp_root)
    for name in ("train", "val", "test"):
        assert loader.get_split(name, balance=False), f"split {name!r} is empty"


def test_ffpp_inconsistent_official_splits_raise(ffpp_root: Path, tmp_path: Path):
    """A corrupt splits file must fail loudly, not silently mis-split."""
    import shutil

    clone = tmp_path / "bad_splits"
    shutil.copytree(ffpp_root, clone)
    # Put identity 000 in test as well as train.
    (clone / "splits" / "test.json").write_text(
        json.dumps([["000", "009"]]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"inconsistent"):
        FFPlusPlusLoader(clone).load_official_splits()


# --------------------------------------------------------------------------- #
# Balance policy (T16)
# --------------------------------------------------------------------------- #
def test_ffpp_train_balances_but_test_does_not(ffpp_root: Path):
    """Balance train; never balance evaluation.

    Downsampling test throws away data, destroys the per-manipulation
    breakdown, and buys nothing -- AUC is prevalence-insensitive.
    """
    loader = FFPlusPlusLoader(ffpp_root)

    train = loader.get_split("train")  # default: balance=True
    n_real = sum(v["label"] == REAL for v in train)
    n_fake = sum(v["label"] == FAKE for v in train)
    assert n_real == n_fake, f"train not balanced: {n_real} real / {n_fake} fake"

    test = loader.get_split("test")  # default: balance=False
    n_real_t = sum(v["label"] == REAL for v in test)
    n_fake_t = sum(v["label"] == FAKE for v in test)
    assert n_fake_t > n_real_t, (
        "test should keep its natural 4-manipulations-to-1-real skew; "
        "balancing it would discard 75% of the fakes"
    )


# --------------------------------------------------------------------------- #
# Celeb-DF loader
# --------------------------------------------------------------------------- #
def test_celebdf_uses_official_testing_list_by_default(celebdf_root: Path):
    loader = CelebDFLoader(celebdf_root)
    videos = loader.get_video_paths()
    listed = loader.load_testing_list()
    assert listed is not None
    assert len(videos) == len(listed)


def test_celebdf_label_polarity_is_not_inverted(celebdf_root: Path):
    """Celeb-DF labels real=1/fake=0; we use real=0/fake=1.

    Getting this backwards silently inverts every metric -- the model would look
    catastrophically bad rather than subtly wrong, but only if you're watching.
    """
    loader = CelebDFLoader(celebdf_root)
    for v in loader.load_testing_list():
        stem_dir = Path(v["path"]).parent.name
        if stem_dir == "Celeb-real":
            assert v["label"] == REAL, f"{v['path']} should be REAL(0)"
        elif stem_dir == "Celeb-synthesis":
            assert v["label"] == FAKE, f"{v['path']} should be FAKE(1)"


def test_celebdf_full_set_is_available_but_opt_in(celebdf_root: Path):
    loader = CelebDFLoader(celebdf_root)
    full = loader.get_video_paths(official_only=False)
    official = loader.get_video_paths(official_only=True)
    assert len(full) > len(official), "full set should be larger than the benchmark"


def test_celebdf_synthesis_keeps_both_identities(celebdf_root: Path):
    loader = CelebDFLoader(celebdf_root)
    fakes = [
        v
        for v in loader.get_video_paths(official_only=False)
        if v["label"] == FAKE
    ]
    assert fakes
    for v in fakes:
        assert len(v["identities"]) == 2, f"fake {v['path']} lost an identity"


def test_celebdf_malformed_testing_list_raises(celebdf_root: Path, tmp_path: Path):
    import shutil

    clone = tmp_path / "bad_list"
    shutil.copytree(celebdf_root, clone)
    (clone / "List_of_testing_videos.txt").write_text(
        "7 Celeb-real/id0_0000.mp4\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"cannot parse"):
        CelebDFLoader(clone).load_testing_list()
