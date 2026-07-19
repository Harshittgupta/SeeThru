"""The dummy dataset must reproduce the REAL datasets' naming traps (T12).

Before T12, create_dummy_dataset.py emitted `person_001_frame_001.jpg` and its
docstring claimed this "matches the identity-aware splitting in
data/dataset_manager.py". It did not. Nothing about FF++'s `<target>_<source>`
convention -- the single most dangerous thing in this pipeline -- was exercised
by any fixture.

These tests assert the fixtures are shaped like the real thing, so T15's split
logic can be tested offline instead of after a 24 GB download and a two-week
EULA wait.
"""

from __future__ import annotations

import json
from pathlib import Path


# --------------------------------------------------------------------------- #
# FaceForensics++
# --------------------------------------------------------------------------- #
def test_ffpp_tree_has_real_layout(ffpp_root: Path):
    """FFPlusPlusLoader walks these exact paths."""
    originals = ffpp_root / "original_sequences" / "youtube" / "c23" / "videos"
    assert originals.is_dir()
    assert len(list(originals.glob("*.mp4"))) == 12

    for manip in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        manip_dir = ffpp_root / "manipulated_sequences" / manip / "c23" / "videos"
        assert manip_dir.is_dir(), f"missing manipulation dir: {manip}"
        assert list(manip_dir.glob("*.mp4")), f"no videos under {manip}"


def test_ffpp_fakes_use_target_source_naming(ffpp_root: Path):
    """Fakes are `<target>_<source>.mp4` -- the two-identity convention."""
    fakes = (ffpp_root / "manipulated_sequences" / "Deepfakes" / "c23" / "videos")
    names = sorted(p.stem for p in fakes.glob("*.mp4"))
    assert names, "no fake videos generated"
    for name in names:
        parts = name.split("_")
        assert len(parts) == 2, f"{name!r} is not <target>_<source>"
        assert all(p.isdigit() for p in parts), f"{name!r} has non-numeric ids"


def test_ffpp_contains_reciprocal_pairs(ffpp_root: Path):
    """The trap must be present, or T15's test proves nothing.

    A reciprocal pair (000_001 and 001_000) is what makes a naive
    `stem.split("_")[0]` splitter provably leak: group on the leading token and
    000_001 keys on '000' while 001_000 keys on '001', so the two can land in
    different splits -- despite both containing BOTH faces.
    """
    fakes = ffpp_root / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    stems = {p.stem for p in fakes.glob("*.mp4")}
    assert "000_001" in stems
    assert "001_000" in stems, "no reciprocal pair -- the leak trap is not reproduced"


def test_ffpp_naive_leading_token_split_provably_leaks(ffpp_root: Path):
    """Demonstrate the actual bug T15 fixes, on real-shaped names.

    This is the executable version of the argument: grouping on the leading
    token puts identity 001 in two different groups at once.
    """
    fakes = ffpp_root / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"

    # The buggy heuristic that video_processor.py:177 currently uses.
    def naive_identity(p: Path) -> str:
        return p.stem.split("_")[0]

    groups: dict[str, set[str]] = {}
    for p in fakes.glob("*.mp4"):
        groups.setdefault(naive_identity(p), set()).add(p.stem)

    # Identity '001' appears as the *source* of 000_001 (grouped under '000')
    # and as the *target* of 001_000 (grouped under '001'). One face, two groups
    # -> a split can separate them -> leak.
    assert "000_001" in groups["000"]
    assert "001_000" in groups["001"]
    assert groups["000"] != groups["001"], (
        "the same face (001) is reachable from two different naive groups; "
        "an identity split over these groups can put it in both train and test"
    )


def test_ffpp_official_splits_exist_and_pair_identities(ffpp_root: Path):
    """Official splits are lists of PAIRS, both members in the same split.

    That property is exactly why T15 uses them instead of a custom splitter.
    """
    splits_dir = ffpp_root / "splits"
    seen: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.json"
        assert path.is_file(), f"missing official split file: {path}"
        pairs = json.loads(path.read_text(encoding="utf-8"))
        assert pairs, f"{name}.json is empty"
        for pair in pairs:
            assert len(pair) == 2, f"{pair!r} is not an identity pair"
            for identity in pair:
                assert identity not in seen or seen[identity] == name, (
                    f"identity {identity!r} appears in both {seen[identity]!r} "
                    f"and {name!r} -- the official splits should never do this"
                )
                seen[identity] = name


# --------------------------------------------------------------------------- #
# Celeb-DF v2
# --------------------------------------------------------------------------- #
def test_celebdf_tree_has_real_layout(celebdf_root: Path):
    """CelebDFLoader walks these exact paths."""
    assert (celebdf_root / "Celeb-real").is_dir()
    assert (celebdf_root / "Celeb-synthesis").is_dir()
    assert (celebdf_root / "YouTube-real").is_dir()
    assert (celebdf_root / "List_of_testing_videos.txt").is_file()


def test_celebdf_fakes_use_two_identity_naming(celebdf_root: Path):
    """`id3_id5_0001.mp4` -- same two-identity trap as FF++, different spelling."""
    stems = {p.stem for p in (celebdf_root / "Celeb-synthesis").glob("*.mp4")}
    assert "id0_id1_0000" in stems
    assert "id1_id0_0000" in stems, "no reciprocal pair -- leak trap not reproduced"
    for stem in stems:
        parts = stem.split("_")
        assert len(parts) == 3, f"{stem!r} is not <target>_<source>_<clip>"


def test_celebdf_testing_list_is_parseable(celebdf_root: Path):
    """The official 518-video benchmark list format: `<label> <path>` per line.

    Note Celeb-DF labels real=1, fake=0 -- the OPPOSITE of this project's
    CLASS_TO_LABEL (real=0, fake=1). T16 must not get that backwards.
    """
    text = (celebdf_root / "List_of_testing_videos.txt").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines

    labels = set()
    for line in lines:
        label, path = line.split(" ", 1)
        assert label in {"0", "1"}, f"bad label in {line!r}"
        assert path.endswith(".mp4")
        labels.add(label)
    assert labels == {"0", "1"}, "testing list must contain both classes"
