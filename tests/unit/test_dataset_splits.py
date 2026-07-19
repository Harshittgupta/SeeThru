"""Split-integrity tests for DeepfakeDataset (BUILD_PLAN T13, T14).

These are the highest-value tests in the repo. Everything downstream -- every
AUC, every claim about generalization -- is meaningless if the splits leak.

The two failure modes they guard:

1. **Collapse**: the identity function returns the same id for everything, so
   there is 1 identity, round(0.15 * 1) == 0, and val/test are silently EMPTY.
   Nothing raises. Early stopping on val loss can never fire.

2. **Explosion**: the identity function returns a unique id per file, so
   "identity-separated" degenerates into a random per-sample split and the same
   subject appears in train and test. Metrics inflate; nothing raises.

Both are silent. That is what makes them expensive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.dataset_manager import DeepfakeDataset, build_splits, default_identity_fn


# --------------------------------------------------------------------------- #
# The identity function itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        # Dummy dataset: identity is person_001, NOT "person". Getting this
        # wrong collapses all 10 identities into 1.
        ("person_001_frame_001", "person_001"),
        ("person_010_frame_004", "person_010"),
        # A video for the same subject must map to the SAME identity as its
        # frames, or a subject splits across train and test.
        ("person_001", "person_001"),
    ],
)
def test_identity_fn_does_not_collapse_dummy_naming(stem: str, expected: str):
    assert default_identity_fn(Path(f"{stem}.jpg")) == expected


def test_identity_fn_distinguishes_dummy_subjects():
    """The property that actually matters: distinct subjects -> distinct ids."""
    ids = {
        default_identity_fn(Path(f"person_{i:03d}_frame_001.jpg"))
        for i in range(1, 11)
    }
    assert len(ids) == 10, f"expected 10 distinct identities, got {len(ids)}: {ids}"


# --------------------------------------------------------------------------- #
# The split contract
# --------------------------------------------------------------------------- #
def test_all_splits_are_non_empty(dummy_images_root: Path):
    """train/val/test must all contain samples.

    This is the test that catches the collapse bug. On a degenerate identity
    map, val and test come back with len == 0 and NO exception.
    """
    splits = build_splits(dummy_images_root)
    for name, ds in splits.items():
        assert len(ds) > 0, f"split {name!r} is empty -- identity split collapsed"


def test_no_identity_leakage(dummy_images_root: Path):
    """No subject may appear in more than one split.

    The single highest-value assertion in this codebase.
    """
    splits = build_splits(dummy_images_root)
    ids = {name: set(ds.identities()) for name, ds in splits.items()}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a] & ids[b]
        assert not overlap, f"identity leak between {a} and {b}: {sorted(overlap)}"


def test_no_sample_appears_in_two_splits(dummy_images_root: Path):
    """Belt and braces: paths must be disjoint too, not just identities."""
    splits = build_splits(dummy_images_root)
    paths = {
        name: {str(p) for p, _label, _identity in ds.samples}
        for name, ds in splits.items()
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (paths[a] & paths[b]), f"sample leak between {a} and {b}"


def test_splits_partition_the_identities(dummy_images_root: Path):
    """Every identity lands in exactly one split -- none dropped, none doubled."""
    everything = DeepfakeDataset(dummy_images_root, split="all", balance=False)
    all_ids = set(everything.identities())

    splits = build_splits(dummy_images_root, balance=False)
    union: set[str] = set()
    for ds in splits.values():
        union |= set(ds.identities())

    assert union == all_ids, f"identities lost in splitting: {all_ids - union}"


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_split_is_deterministic_for_a_seed(dummy_images_root: Path):
    """Same seed -> identical splits. Underpins the reproducibility claim in
    dataset_manager's module docstring."""
    a = build_splits(dummy_images_root, seed=7)
    b = build_splits(dummy_images_root, seed=7)
    for name in ("train", "val", "test"):
        assert [str(p) for p, _, _ in a[name].samples] == [
            str(p) for p, _, _ in b[name].samples
        ], f"split {name!r} is not deterministic under a fixed seed"


def test_different_seeds_give_different_splits(dummy_images_root: Path):
    """A seed that doesn't change anything is a seed that isn't wired up."""
    a = build_splits(dummy_images_root, seed=1)
    b = build_splits(dummy_images_root, seed=999)
    same = all(
        [str(p) for p, _, _ in a[name].samples] == [str(p) for p, _, _ in b[name].samples]
        for name in ("train", "val", "test")
    )
    assert not same, "seed appears to have no effect on the identity split"


# --------------------------------------------------------------------------- #
# Balance
# --------------------------------------------------------------------------- #
def test_train_split_is_balanced(dummy_images_root: Path):
    """balance=True must give 50:50 real/fake within the split."""
    train = DeepfakeDataset(dummy_images_root, split="train", balance=True)
    counts = train.class_counts()
    assert counts["real"] == counts["fake"], f"train not 50:50: {counts}"


# --------------------------------------------------------------------------- #
# The guards themselves (T11)
#
# A validation nobody tests is a validation that quietly rots. These assert the
# guards *fire*, and that their messages name the actual problem -- an error that
# says "IndexError" teaches nobody anything at 2am.
# --------------------------------------------------------------------------- #
def test_collapsed_identity_fn_raises(dummy_images_root: Path):
    """One identity for everything must raise, not silently empty val/test."""
    with pytest.raises(ValueError, match=r"only 1 distinct identity"):
        DeepfakeDataset(dummy_images_root, split="train", identity_fn=lambda p: "same")


def test_exploded_identity_fn_raises(dummy_images_root: Path):
    """A unique identity per file must raise: that is a random split wearing an
    identity-split costume.

    Uses the full path, not the stem: the dummy set deliberately reuses the same
    filenames under real/ and fake/, so `p.stem` actually *groups* each real/fake
    pair (40 ids for 80 samples). Only the full path is unique per sample.
    """
    with pytest.raises(ValueError, match=r"unique identity for every one"):
        DeepfakeDataset(dummy_images_root, split="train", identity_fn=lambda p: str(p))


def test_stem_identity_fn_groups_real_fake_pairs(dummy_images_root: Path):
    """Sanity check on the above: stems group, so they must NOT trip the guard.

    Guards that fire on correct input are worse than no guard -- people disable
    them.
    """
    ds = DeepfakeDataset(
        dummy_images_root, split="train", balance=False, identity_fn=lambda p: p.stem
    )
    assert len(ds) > 0


def test_collapse_error_message_is_actionable(dummy_images_root: Path):
    """The message must show the mapping that caused it."""
    with pytest.raises(ValueError) as exc:
        DeepfakeDataset(dummy_images_root, split="train", identity_fn=lambda p: "x")
    msg = str(exc.value)
    assert "val/test would be EMPTY" in msg
    assert "Example mapping" in msg
    assert "identity_fn" in msg


def test_empty_split_raises_rather_than_returning_zero(dummy_images_root: Path):
    """Ratios that starve a split must raise.

    A DataLoader over an empty dataset yields no batches, so training would
    "succeed" having never validated once. len(ds) == 0 is not an acceptable
    answer to "give me the val split".
    """
    with pytest.raises(ValueError, match=r"split 'test' is empty"):
        DeepfakeDataset(
            dummy_images_root,
            split="test",
            # 10 identities: train=10, val=0, test=0.
            split_ratios=(1.0, 0.0, 0.0),
        )
