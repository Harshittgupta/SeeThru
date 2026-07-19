"""Tests for the split auditor (BUILD_PLAN T17).

The auditor is the thing that catches leakage in CI, so its own failure modes
matter more than most. The tests that count are the ones proving it FAILS when it
should -- an auditor that always passes is worse than no auditor, because it
launders a broken split as a verified one.
"""

from __future__ import annotations

from pathlib import Path

from data.audit_splits import (
    audit_ffpp,
    audit_images,
    audit_split_mapping,
    run_audits,
    split_fingerprint,
)

REAL, FAKE = 0, 1


def _video(path: str, label: int, identities: tuple, group: str) -> dict:
    return {"path": path, "label": label, "identities": identities, "identity": group}


def _clean_splits() -> dict:
    return {
        "train": [
            _video("a.mp4", REAL, ("000",), "000"),
            _video("b.mp4", FAKE, ("000", "001"), "000"),
        ],
        "val": [
            _video("c.mp4", REAL, ("004",), "004"),
            _video("d.mp4", FAKE, ("004", "005"), "004"),
        ],
        "test": [
            _video("e.mp4", REAL, ("008",), "008"),
            _video("f.mp4", FAKE, ("008", "009"), "008"),
        ],
    }


def test_clean_splits_pass():
    """Baseline: the auditor must not cry wolf on a correct split."""
    assert audit_split_mapping("clean", _clean_splits()).ok


def test_detects_identity_leak_via_second_id():
    """THE case: a fake's SOURCE identity also appears in another split.

    The leading-token bug (T15) produces exactly this -- and note that the
    train-side video's *group* key is '000', so an auditor that checked
    video["identity"] instead of video["identities"] would miss it entirely.
    """
    splits = _clean_splits()
    # 009 is the source of test's f.mp4, and now also appears in train.
    splits["train"].append(_video("g.mp4", FAKE, ("000", "009"), "000"))

    audit = audit_split_mapping("leaky", splits)
    assert not audit.ok
    assert any("IDENTITY LEAK" in f for f in audit.failures)
    assert any("009" in f for f in audit.failures)


def test_detects_sample_leak():
    splits = _clean_splits()
    splits["test"].append(_video("a.mp4", REAL, ("000",), "000"))  # same path as train
    audit = audit_split_mapping("dup", splits)
    assert not audit.ok
    assert any("SAMPLE LEAK" in f for f in audit.failures)


def test_detects_group_straddle():
    """A group in two splits means a swap chain got cut."""
    splits = _clean_splits()
    splits["val"].append(_video("h.mp4", FAKE, ("020", "021"), "000"))  # group 000
    audit = audit_split_mapping("straddle", splits)
    assert not audit.ok
    assert any("GROUP STRADDLE" in f for f in audit.failures)


def test_empty_split_fails_and_short_circuits():
    """Empty splits must FAIL, and must not be reported as 'no leaks found'.

    This is the false-green that started all of this: empty sets do not
    intersect, so every leak check passes over them.
    """
    splits = _clean_splits()
    splits["val"] = []
    audit = audit_split_mapping("empty", splits)
    assert not audit.ok
    assert any("EMPTY" in f for f in audit.failures)
    # It must not also claim the leak checks passed.
    assert any("vacuously" in f for f in audit.failures)


def test_warns_when_eval_split_is_balanced():
    """A 50:50 test split means someone balanced it -- discarding data for nothing."""
    splits = _clean_splits()  # val/test are 1 real / 1 fake => exactly 50:50
    audit = audit_split_mapping("balanced-eval", splits)
    assert any("50:50" in w for w in audit.warnings)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def test_fingerprint_is_stable_and_order_independent():
    a = split_fingerprint(_clean_splits())
    shuffled = _clean_splits()
    shuffled["train"].reverse()
    assert a == split_fingerprint(shuffled)


def test_fingerprint_changes_when_the_split_changes():
    """If this were insensitive, 'did the split change?' would be unanswerable."""
    a = split_fingerprint(_clean_splits())
    moved = _clean_splits()
    moved["test"].append(moved["train"].pop())
    assert a != split_fingerprint(moved)


# --------------------------------------------------------------------------- #
# End-to-end against the real fixtures
# --------------------------------------------------------------------------- #
def test_audit_ffpp_passes_on_fixed_loader(ffpp_root: Path):
    """The loaders are fixed (T15), so the auditor should now pass on them."""
    audit = audit_ffpp(ffpp_root)
    assert audit.ok, f"audit failed: {audit.failures}"


def test_audit_images_passes(dummy_images_root: Path):
    audit = audit_images(dummy_images_root)
    assert audit.ok, f"audit failed: {audit.failures}"


def test_run_audits_exit_codes(dummy_images_root: Path, ffpp_root: Path):
    """Exit code is the contract: CI depends on non-zero meaning 'do not train'."""
    assert run_audits(images=dummy_images_root, ffpp=ffpp_root) == 0
    assert run_audits() == 2  # nothing requested
