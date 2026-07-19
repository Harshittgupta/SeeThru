"""Metrics tests (BUILD_PLAN T28).

Two tests here are worth more than the rest combined:

* `test_select_threshold_refuses_test_split` -- threshold selection on test is
  the most common way a project reports a number that doesn't survive contact
  with reality, and it is *invisible* in the output. The number just comes out
  better. The guard has to be structural.

* `test_accuracy_is_misleading_on_skewed_data` -- executable proof of why AUC is
  the headline: an always-"fake" model scores 86.4% accuracy on Celeb-DF's real
  prior, and 0.5 AUC. One of those numbers tells the truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.utils.metrics import (
    aggregate_frames_to_video,
    compute_eer,
    compute_metrics,
    per_manipulation_breakdown,
    select_threshold,
)

REAL, FAKE = 0, 1


# --------------------------------------------------------------------------- #
# Why AUC and not accuracy
# --------------------------------------------------------------------------- #
def test_accuracy_is_misleading_on_skewed_data():
    """The Celeb-DF trap, made executable.

    Celeb-DF's full corpus is 890 real / 5639 fake = 86.4% fake. A model that
    always answers "fake" scores 86.4% accuracy -- and 0.5 AUC. This is exactly
    why T16 filters to the official 518-video subset and why AUC is the headline.
    """
    y_true = np.array([REAL] * 890 + [FAKE] * 5639)
    always_fake = np.ones(len(y_true))  # p(fake) = 1.0 for everything

    m = compute_metrics(y_true, always_fake)
    assert m.accuracy == pytest.approx(0.864, abs=0.001), "the flattering number"
    assert m.auc == pytest.approx(0.5, abs=0.001), "the honest one"


def test_auc_is_prevalence_insensitive():
    """The property that makes AUC safe to compare across splits with different
    priors -- which is why we deliberately do NOT balance val/test (T16)."""
    rng = np.random.default_rng(0)
    real = rng.normal(0.3, 0.1, 500)
    fake = rng.normal(0.7, 0.1, 500)

    balanced = compute_metrics([REAL] * 500 + [FAKE] * 500, np.r_[real, fake])
    skewed = compute_metrics([REAL] * 50 + [FAKE] * 500, np.r_[real[:50], fake])
    assert balanced.auc == pytest.approx(skewed.auc, abs=0.03)


# --------------------------------------------------------------------------- #
# The threshold rule
# --------------------------------------------------------------------------- #
def test_select_threshold_refuses_test_split():
    """Structural guard. Tuning on test is invisible in the output."""
    y_true = [REAL, REAL, FAKE, FAKE]
    y_score = [0.1, 0.2, 0.8, 0.9]

    assert not np.isnan(select_threshold(y_true, y_score, split="val"))
    for split in ("test", "train", "celebdf_test"):
        with pytest.raises(ValueError, match=r"must be chosen on val and FROZEN"):
            select_threshold(y_true, y_score, split=split)


def test_frozen_threshold_is_honoured_on_test():
    """compute_metrics(threshold=...) must use the given value, not re-derive it."""
    y_true = np.array([REAL] * 50 + [FAKE] * 50)
    y_score = np.r_[np.full(50, 0.2), np.full(50, 0.8)]

    generous = compute_metrics(y_true, y_score, threshold=0.99)  # nothing is fake
    assert generous.confusion["tp"] == 0
    assert generous.confusion["fn"] == 50


# --------------------------------------------------------------------------- #
# EER
# --------------------------------------------------------------------------- #
def test_eer_of_a_perfect_classifier_is_zero():
    y_true = np.array([REAL] * 50 + [FAKE] * 50)
    y_score = np.r_[np.zeros(50), np.ones(50)]
    eer, _ = compute_eer(y_true, y_score)
    assert eer == pytest.approx(0.0, abs=1e-6)


def test_eer_of_a_coin_flip_is_about_half():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 2000)
    eer, _ = compute_eer(y_true, rng.random(2000))
    assert eer == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# Degenerate input -- must not kill a run
# --------------------------------------------------------------------------- #
def test_single_class_returns_nan_rather_than_raising():
    """A val split that happens to be single-class should show up as NaN in the
    log, not blow up the run at epoch 7. sklearn raises here; we don't."""
    m = compute_metrics([FAKE] * 10, np.full(10, 0.9))
    assert np.isnan(m.auc)
    assert np.isnan(m.eer)
    assert not np.isnan(m.accuracy)  # still computable


def test_empty_input_is_survivable():
    m = compute_metrics([], [])
    assert m.n == 0
    assert np.isnan(m.auc)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match=r"shape mismatch"):
        compute_metrics([0, 1], [0.5])


# --------------------------------------------------------------------------- #
# Per-manipulation breakdown -- the finding a single AUC hides
# --------------------------------------------------------------------------- #
def test_per_manipulation_isolates_a_weak_method():
    """Simulates the real result: three strong methods and NeuralTextures weak.

    The averaged AUC looks fine while one method is near chance. That gap is the
    most interesting thing this project will measure, and it only exists if
    `manipulation` survives to here (T19).
    """
    rng = np.random.default_rng(0)
    n = 200
    y_true = np.r_[np.zeros(n, int), np.ones(3 * n, int)]
    scores = np.r_[
        rng.normal(0.2, 0.1, n),        # real
        rng.normal(0.9, 0.05, n),       # Deepfakes -- easy
        rng.normal(0.9, 0.05, n),       # FaceSwap -- easy
        rng.normal(0.35, 0.2, n),       # NeuralTextures -- hard (mouth only)
    ]
    manips = np.array(["none"] * n + ["Deepfakes"] * n + ["FaceSwap"] * n + ["NeuralTextures"] * n)

    out = per_manipulation_breakdown(y_true, scores, manips, threshold=0.5)
    assert out["Deepfakes"]["auc"] > 0.99
    assert out["FaceSwap"]["auc"] > 0.99
    assert out["NeuralTextures"]["auc"] < out["Deepfakes"]["auc"]
    assert "none" not in out, "real is the comparison set, not a method"


def test_per_manipulation_compares_against_the_shared_real_set():
    """Each method is scored against the SAME reals, or the numbers aren't
    comparable -- and a groupby over fakes alone gives one class and no AUC."""
    y_true = np.array([REAL, REAL, FAKE, FAKE])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    manips = np.array(["none", "none", "Deepfakes", "FaceSwap"])

    out = per_manipulation_breakdown(y_true, y_score, manips)
    assert set(out) == {"Deepfakes", "FaceSwap"}
    for entry in out.values():
        assert "auc" in entry, "each method needs the reals to have an AUC at all"


# --------------------------------------------------------------------------- #
# Frame -> video aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_uses_mean_not_max():
    """Mean, not max: max is a max over noise, so one bad frame flips a video."""
    ids = ["v1"] * 4
    scores = [0.1, 0.1, 0.1, 0.9]  # one outlier frame
    _vids, agg, _labels = aggregate_frames_to_video(ids, scores, [FAKE] * 4)
    assert agg[0] == pytest.approx(0.3)  # mean; max would be 0.9


def test_aggregate_groups_and_sorts():
    ids = ["b", "a", "b", "a"]
    _vids, agg, labels = aggregate_frames_to_video(ids, [0.2, 0.8, 0.4, 0.6], [FAKE, REAL, FAKE, REAL])
    vids, _, _ = aggregate_frames_to_video(ids, [0.2, 0.8, 0.4, 0.6], [FAKE, REAL, FAKE, REAL])
    assert vids == ["a", "b"]  # deterministic order
    assert agg[0] == pytest.approx(0.7)  # a: (0.8+0.6)/2
    assert agg[1] == pytest.approx(0.3)  # b: (0.2+0.4)/2
    assert labels.tolist() == [REAL, FAKE]


def test_aggregate_rejects_conflicting_labels():
    """Frames of one video disagreeing on the label means the manifest is wrong."""
    with pytest.raises(ValueError, match=r"conflicting labels"):
        aggregate_frames_to_video(["v1", "v1"], [0.5, 0.5], [REAL, FAKE])


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_metrics_dict_is_checkpoint_safe():
    """Metrics ride along in the checkpoint, so no numpy scalars allowed (T24)."""
    m = compute_metrics([REAL, FAKE], [0.2, 0.8], loss=0.3)
    d = m.to_dict()

    def assert_primitive(obj, path="metrics"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert_primitive(v, f"{path}[{k!r}]")
        else:
            assert isinstance(obj, (int, float, str, bool)), (
                f"{path} is {type(obj).__name__}, which weights_only=True cannot load"
            )

    assert_primitive(d)
