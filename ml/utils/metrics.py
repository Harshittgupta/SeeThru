"""Evaluation metrics for SEETHRU (BUILD_PLAN T28).

**Why accuracy is the wrong headline**, spelled out because the spec asks for it
and it will mislead you:

1. It bakes in a threshold. ``argmax`` freezes the decision at p=0.5, but the
   product needs a tunable operating point, and 0.5 is almost never the right
   one for a detector.
2. It depends on the class prior. FF++ has 4 fake manipulations per real video,
   so a model that always says "fake" scores 80%. On Celeb-DF's full corpus that
   trick scores 86.4%.
3. It hides the finding. Deepfakes/Face2Face/FaceSwap all land ~0.98-0.99 while
   NeuralTextures sits at ~0.90-0.95 -- it only edits the mouth. One averaged
   number erases the most interesting thing you will learn.

So the headline is **AUC-ROC** (threshold-free, prevalence-insensitive), reported
next to **AP**, **EER**, and a **per-manipulation breakdown**.

**The threshold rule:** pick it on val, freeze it, apply it everywhere else.
Re-tuning on test is the single most common way projects report a number that
does not survive contact with reality — and it is invisible in the output, which
is why `select_threshold` lives here and takes `split` as an argument it checks.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

REAL, FAKE = 0, 1


@dataclass
class Metrics:
    """Computed metrics for one split. ``fake`` is the positive class."""

    n: int
    n_real: int
    n_fake: int
    loss: float = float("nan")
    auc: float = float("nan")
    ap: float = float("nan")
    eer: float = float("nan")
    eer_threshold: float = float("nan")
    accuracy: float = float("nan")  # at 0.5; reported, never headlined
    accuracy_at_eer: float = float("nan")
    confusion: dict = field(default_factory=dict)
    per_manipulation: dict = field(default_factory=dict)

    # Raw per-sample arrays, kept so a caller can select a threshold via
    # select_threshold() (whose split guard is the point) rather than reading
    # `eer_threshold` off whichever split happens to be to hand. Deliberately
    # excluded from to_dict(): they are large, and a checkpoint holding a copy of
    # the test scores is asking for someone to tune against them later.
    y_true: list = field(default_factory=list, repr=False)
    y_score: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        """Primitive-only, so it can go straight into a checkpoint (T24)."""
        return {
            "n": int(self.n),
            "n_real": int(self.n_real),
            "n_fake": int(self.n_fake),
            "loss": float(self.loss),
            "auc": float(self.auc),
            "ap": float(self.ap),
            "eer": float(self.eer),
            "eer_threshold": float(self.eer_threshold),
            "accuracy": float(self.accuracy),
            "accuracy_at_eer": float(self.accuracy_at_eer),
            "confusion": {k: int(v) for k, v in self.confusion.items()},
            "per_manipulation": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in self.per_manipulation.items()
            },
        }

    def summary(self) -> str:
        """One line for the log. AUC first, because AUC is the headline."""
        return (
            f"auc={self.auc:.4f} ap={self.ap:.4f} eer={self.eer:.4f} "
            f"loss={self.loss:.4f} acc@0.5={self.accuracy:.4f} "
            f"(n={self.n}, {self.n_real}r/{self.n_fake}f)"
        )


def _degenerate(y_true: np.ndarray) -> bool:
    """True when only one class is present, so AUC/AP/EER are undefined.

    Returning NaN beats sklearn's exception here: a val split that happens to be
    single-class should not kill a training run at epoch 7, it should be visible
    as NaN in the log.
    """
    return len(np.unique(y_true)) < 2


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold achieving it.

    EER is where FPR == FNR. It is a single prior-free number that summarises the
    whole ROC, and its threshold is a defensible default operating point when you
    have no cost asymmetry to justify anything else.
    """
    from sklearn.metrics import roc_curve

    if _degenerate(y_true):
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    loss: float = float("nan"),
    manipulations: Sequence[str] | None = None,
    threshold: float | None = None,
) -> Metrics:
    """Full metric set for one split.

    Args:
        y_true: 0=real, 1=fake.
        y_score: P(fake). NOT logits -- ROC needs a monotone score, and mixing
            the two silently across calls makes thresholds meaningless.
        loss: Mean loss, if the caller tracked it.
        manipulations: Per-sample method name ("Deepfakes", ..., "none" for real).
            Enables the per-method breakdown -- the thing a single AUC hides.
        threshold: Operating point for accuracy_at_eer. **Pass val's frozen
            threshold when scoring test**; if None, it is derived from these data,
            which is only legitimate on val.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")

    n_fake = int((y_true == FAKE).sum())
    m = Metrics(
        n=len(y_true),
        n_real=len(y_true) - n_fake,
        n_fake=n_fake,
        loss=float(loss),
        y_true=y_true.tolist(),
        y_score=y_score.tolist(),
    )

    if len(y_true) == 0:
        return m

    if _degenerate(y_true):
        logger.warning(
            "Only one class present (%d samples, all %s) -- AUC/AP/EER are "
            "undefined and reported as NaN.",
            len(y_true),
            "fake" if n_fake else "real",
        )
    else:
        m.auc = float(roc_auc_score(y_true, y_score))
        m.ap = float(average_precision_score(y_true, y_score))
        m.eer, m.eer_threshold = compute_eer(y_true, y_score)

    m.accuracy = float(((y_score >= 0.5).astype(int) == y_true).mean())

    op = threshold if threshold is not None else m.eer_threshold
    if not np.isnan(op):
        pred = (y_score >= op).astype(int)
        m.accuracy_at_eer = float((pred == y_true).mean())
        m.confusion = {
            "tn": int(((pred == REAL) & (y_true == REAL)).sum()),
            "fp": int(((pred == FAKE) & (y_true == REAL)).sum()),
            "fn": int(((pred == REAL) & (y_true == FAKE)).sum()),
            "tp": int(((pred == FAKE) & (y_true == FAKE)).sum()),
        }

    if manipulations is not None:
        m.per_manipulation = per_manipulation_breakdown(
            y_true, y_score, manipulations, threshold=op
        )
    return m


def per_manipulation_breakdown(
    y_true: np.ndarray,
    y_score: np.ndarray,
    manipulations: Sequence[str],
    threshold: float = float("nan"),
) -> dict:
    """Per-method AUC/recall, each measured against the SAME real set.

    Comparing each fake method against the shared real pool is what makes the
    numbers comparable to each other -- score a method against only its own fakes
    and AUC is undefined (one class), which is why this is not just a groupby.

    Expect: Deepfakes/Face2Face/FaceSwap ~0.98-0.99, **NeuralTextures ~0.90-0.95**
    (it only edits the mouth region). If NeuralTextures is not your worst method,
    something is unusual -- check the split before celebrating.
    """
    from sklearn.metrics import roc_auc_score

    manipulations = np.asarray(manipulations, dtype=object)
    real_mask = y_true == REAL
    out: dict = {}

    for method in sorted({m for m in manipulations if m and m != "none"}):
        method_mask = (manipulations == method) & (y_true == FAKE)
        n_method = int(method_mask.sum())
        if n_method == 0:
            continue

        mask = real_mask | method_mask
        sub_true, sub_score = y_true[mask], y_score[mask]

        entry = {"n": float(n_method)}
        if not _degenerate(sub_true):
            entry["auc"] = float(roc_auc_score(sub_true, sub_score))
        if not np.isnan(threshold):
            entry["recall"] = float((y_score[method_mask] >= threshold).mean())
        out[method] = entry
    return out


def select_threshold(
    y_true: Sequence[int], y_score: Sequence[float], split: str = "val"
) -> float:
    """Choose the operating point. **On val only.**

    Raises on any other split. That is the entire reason this function exists
    rather than being an inline call: threshold selection on test is invisible in
    the output — the number just comes out better — so the guard has to be
    structural rather than a comment asking people to be careful.
    """
    if split != "val":
        raise ValueError(
            f"select_threshold called on split={split!r}. The operating point "
            f"must be chosen on val and FROZEN, then applied unchanged to test "
            f"and to cross-dataset eval. Tuning it on test reports a number that "
            f"does not exist in production, and nothing in the output reveals it."
        )
    _eer, threshold = compute_eer(np.asarray(y_true), np.asarray(y_score))
    return threshold


def aggregate_frames_to_video(
    video_ids: Sequence[str], y_score: Sequence[float], y_true: Sequence[int]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pool frame scores into video scores by **mean of probabilities**.

    Mean, not max: max is a max over noise, so one bad frame flips a whole video
    and the metric becomes a function of the worst outlier. Mean-of-logits is the
    other common choice; mean-of-probabilities is more robust to a single
    saturated frame.

    Returns ``(video_ids, scores, labels)``, sorted by id for determinism.
    """
    y_score = np.asarray(y_score, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    buckets: dict[str, list[int]] = {}
    for i, vid in enumerate(video_ids):
        buckets.setdefault(vid, []).append(i)

    ids = sorted(buckets)
    scores = np.array([y_score[buckets[v]].mean() for v in ids])
    labels = np.array([y_true[buckets[v]][0] for v in ids])

    for vid in ids:
        idx = buckets[vid]
        if len(set(y_true[idx].tolist())) > 1:
            raise ValueError(f"video {vid!r} has frames with conflicting labels")
    return ids, scores, labels
