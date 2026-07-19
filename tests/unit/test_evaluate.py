"""Evaluation tests (BUILD_PLAN T35).

Most of this file tests `sanity_commentary`, which is the unusual part of
evaluate.py and the part worth defending. Every failure mode in this project
makes numbers look BETTER -- a leak does not raise, it hands you 0.98
cross-dataset AUC and lets you write it up. So the report has to be suspicious on
the reader's behalf, and these tests make sure it actually is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.evaluate import evaluate_checkpoint, sanity_commentary


def _results(**over) -> dict:
    base = {
        "threshold": 0.5,
        "calibrated": True,
        "val": {"auc": 0.99, "ap": 0.99, "eer": 0.03, "accuracy_at_eer": 0.97, "n": 100},
        "test": {
            "auc": 0.98, "ap": 0.98, "eer": 0.05, "accuracy_at_eer": 0.95, "n": 100,
            "per_manipulation": {},
        },
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# The guards
# --------------------------------------------------------------------------- #
def test_flags_impossibly_good_cross_dataset_auc():
    """0.98 cross-dataset is a leak, not a breakthrough.

    The single most valuable line in the report. Published Celeb-DF numbers sit
    at 0.65-0.75; the best specialised methods reach ~0.93.
    """
    notes = sanity_commentary(_results(cross_dataset={"auc": 0.98}))
    assert any("leak" in n for n in notes)
    assert any("audit_splits" in n for n in notes), "must say what to actually run"


def test_flags_near_chance_cross_dataset_as_possible_label_inversion():
    """AUC ~0.3 means inverted labels far more often than a bad model.

    Celeb-DF's list encodes real=1/fake=0, the inverse of ours -- so this is a
    live trap, not a hypothetical (T16).
    """
    notes = sanity_commentary(_results(cross_dataset={"auc": 0.28}))
    assert any("polarity" in n for n in notes)
    assert any("0.72" in n for n in notes), "should show 1-AUC as the likely true score"


def test_reassures_on_a_normal_cross_dataset_result():
    """0.70 is a GOOD result and must not read as a failure.

    Without this, the expected ~30-point drop from in-domain looks like something
    went wrong, and someone goes hunting for a bug that does not exist.
    """
    notes = sanity_commentary(_results(cross_dataset={"auc": 0.70}))
    assert any("NORMAL" in n and "0.65-0.75" in n for n in notes)


def test_flags_a_suspiciously_small_generalization_gap():
    """Every published FF++ detector drops ~0.25-0.30 cross-dataset."""
    notes = sanity_commentary(_results(cross_dataset={"auc": 0.96}))
    assert any("differ by only" in n or "leak" in n for n in notes)


def test_flags_uncalibrated_checkpoints():
    """No calibration exists yet (T78), so this fires on every current run."""
    notes = sanity_commentary(_results(calibrated=False))
    assert any("NOT calibrated" in n for n in notes)


def test_quiet_when_calibrated_and_in_band():
    notes = sanity_commentary(_results(calibrated=True, cross_dataset={"auc": 0.70}))
    assert not any("NOT calibrated" in n for n in notes)


def test_flags_an_unexpected_worst_method():
    """NeuralTextures edits only the mouth and is reliably the weakest.

    If something else is worse, the split is worth checking before reporting.
    """
    results = _results()
    results["test"]["per_manipulation"] = {
        "NeuralTextures": {"auc": 0.95, "n": 100},
        "Deepfakes": {"auc": 0.71, "n": 100},  # unexpectedly bad
    }
    notes = sanity_commentary(results)
    assert any("Deepfakes" in n and "WEAKEST" in n for n in notes)


def test_no_false_alarm_when_neuraltextures_is_worst():
    """A guard that fires on the expected result gets ignored, then deleted."""
    results = _results()
    results["test"]["per_manipulation"] = {
        "NeuralTextures": {"auc": 0.92, "n": 100},
        "Deepfakes": {"auc": 0.99, "n": 100},
    }
    assert not any("WEAKEST" in n for n in sanity_commentary(results))


def test_handles_missing_cross_dataset():
    """Must not crash when only in-domain was run."""
    assert isinstance(sanity_commentary(_results()), list)


def test_handles_nan_auc():
    """A single-class split yields NaN AUC; the commentary must survive it."""
    assert isinstance(
        sanity_commentary(_results(cross_dataset={"auc": float("nan")})), list
    )


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_evaluate_checkpoint_end_to_end(tmp_path: Path, dummy_images_root: Path, image_model):
    """Full path: save a checkpoint, evaluate it, get results.json.

    Proves the threshold flows val -> test rather than being re-derived.
    """
    from ml.checkpoint import save
    from ml.config import Config

    cfg = Config.from_dict(
        {
            "data": {"root": str(dummy_images_root), "num_workers": 0, "balance": False},
            "model": {"pretrained": False},
            "train": {"batch_size": 4, "amp": False, "device": "cpu", "channels_last": False},
        }
    )
    path = save(tmp_path / "best.pt", image_model, epoch=0, config=cfg.to_checkpoint_dict())

    results = evaluate_checkpoint(path, data_root=str(dummy_images_root), device_spec="cpu")

    assert "val" in results and "test" in results
    assert "threshold" in results
    assert (tmp_path / "results.json").is_file()
    # The frozen val threshold, not one re-derived from test.
    assert results["threshold"] == pytest.approx(results["val"]["eer_threshold"], abs=1e-9)


def test_threshold_guard_is_in_the_real_path():
    """evaluate.py must route through select_threshold, not read eer_threshold.

    Every Metrics carries an eer_threshold computed from its own split, so
    reading one off `test` would be a one-character mistake that nothing catches.
    """
    import inspect

    from ml import evaluate as module

    src = inspect.getsource(module.evaluate_checkpoint)
    assert 'select_threshold(' in src and 'split="val"' in src, (
        "evaluate_checkpoint must select the threshold via select_threshold(split='val') "
        "so the guard applies in production, not only in tests"
    )
