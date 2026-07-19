"""The explanation contract (BUILD_PLAN T46).

What crosses the boundary from `ml/` to the backend (T58) and on to the UI (T62).

Two rules:

**1. `to_dict()` returns JSON-safe scalars only.** No numpy arrays, no tensors, no
Figures. The backend serialises this straight to JSON, and a stray `np.float32`
raises there rather than here -- at request time, in production, where the
traceback is least useful.

**2. Images are NOT in the dict.** They travel separately as raw PNG bytes
(:class:`ExplanationArtifacts`). A video explanation carries 16 CAM overlays plus
a spectrum plus the evolution strip; base64-ing those into the JSON is ~1.6 MB of
text that must be fully buffered and parsed before the verdict can be shown, and
cannot be cached. The backend writes them to an artifact store and returns URLs.

**On honesty.** Several fields exist purely to stop the UI overclaiming, and they
are not optional decoration:

* ``calibrated`` -- false today and for the foreseeable future (no calibration
  code exists; T78). An uncalibrated softmax is NOT a probability, so the UI must
  suppress the percentage (T63).
* ``verdict`` includes ``uncertain`` -- forcing a binary answer on a near-0.5
  margin is the most misleading thing this system could do.
* ``FrameScore.interpolated`` -- a copied frame is not an observation. Plotting it
  on a timeline as a measured point is simply false.
* ``degenerate`` -- names any component that failed. A dead heatmap must be
  reported as dead, never rendered as a confident-looking map (T53).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# |p_fake - 0.5| below this is reported as `uncertain` rather than forced to a
# side. Deliberately wide: an uncalibrated model's probabilities are not
# meaningful near the boundary.
UNCERTAIN_MARGIN = 0.10

BRANCHES = ("spatial", "frequency", "temporal")


@dataclass(frozen=True)
class BranchAttribution:
    """One branch's causal contribution, measured by ablation (ADR 0001).

    ``delta`` is the change in the fake logit when this branch is replaced by its
    training-set mean. It is a **measured effect, not a share**: the three deltas
    do NOT sum to anything in particular, and the UI must not render them as a
    pie chart or a normalised stacked bar (T62).

    That non-normalisation is a feature. AttentionFusion's softmax weights are
    forced to sum to 1, so they literally cannot express "all three branches
    agree strongly" or "none of them are driving this" -- both of which are
    informative states, and both of which ablation reports naturally.
    """

    branch: str
    delta: float
    #: What the branch was replaced with. "mean" is correct; "zero" is
    #: off-manifold (the MLP never saw a zero vector in training) and makes part
    #: of the measured delta an artifact of the ablation itself.
    baseline: str = "mean"

    def to_dict(self) -> dict:
        return {"branch": self.branch, "delta": float(self.delta), "baseline": self.baseline}


@dataclass(frozen=True)
class FrequencyEvidence:
    """What the frequency branch saw (T49)."""

    #: Fraction of log-spectrum energy above the half-Nyquist radius. One number,
    #: one sentence -- the part a non-expert can actually act on.
    hf_energy_ratio: float
    #: Azimuthally-averaged log-magnitude, low->high frequency. Meaningless on its
    #: own; the UI plots it against real/fake reference bands.
    radial_profile: list[float] = field(default_factory=list)
    #: Reference means over the training set, if available.
    reference_real: list[float] = field(default_factory=list)
    reference_fake: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hf_energy_ratio": float(self.hf_energy_ratio),
            "radial_profile": [float(v) for v in self.radial_profile],
            "reference_real": [float(v) for v in self.reference_real],
            "reference_fake": [float(v) for v in self.reference_fake],
        }


@dataclass(frozen=True)
class FrameScore:
    """One sampled frame on the manipulation timeline (T50)."""

    index: int              # position within the clip, 0..T-1
    source_index: int       # true frame index in the original video
    p_fake: float
    t_seconds: float | None = None
    #: Raw softmax-over-time attention. At T=16 uniform is 0.0625, so this is
    #: NOT comparable to a 0-1 threshold -- see `attention_norm`.
    attention: float | None = None
    #: attention / max(attention). This is what a 0.6 threshold can act on; the
    #: spec's raw 0.6 could never fire (T50).
    attention_norm: float | None = None
    suspicious: bool = False
    #: True when this frame's face was copied from a neighbour (detection failed)
    #: or duplicated as padding. NOT an observation; must not be drawn as one.
    interpolated: bool = False

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "source_index": int(self.source_index),
            "p_fake": float(self.p_fake),
            "t_seconds": None if self.t_seconds is None else float(self.t_seconds),
            "attention": None if self.attention is None else float(self.attention),
            "attention_norm": (
                None if self.attention_norm is None else float(self.attention_norm)
            ),
            "suspicious": bool(self.suspicious),
            "interpolated": bool(self.interpolated),
        }


@dataclass(frozen=True)
class TimelineSpan:
    """A merged run of suspicious frames.

    Only emitted when >=2 *consecutive* sampled frames clear the threshold. With
    16 samples spread over a whole video, adjacent samples can be ~19 s apart, so
    a span is a "sampled region", not a continuous detection (T62).
    """

    start_s: float
    end_s: float
    mean_p_fake: float
    n_frames: int

    def to_dict(self) -> dict:
        return {
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "mean_p_fake": float(self.mean_p_fake),
            "n_frames": int(self.n_frames),
        }


@dataclass(frozen=True)
class Explanation:
    """The full explanation for one prediction."""

    label: str                    # "real" | "fake"
    verdict: str                  # "real" | "fake" | "uncertain"
    p_fake: float
    calibrated: bool = False

    attribution: list[BranchAttribution] = field(default_factory=list)
    frequency: FrequencyEvidence | None = None
    timeline: list[FrameScore] = field(default_factory=list)
    spans: list[TimelineSpan] = field(default_factory=list)

    #: Component -> did it fail? A degenerate heatmap must be reported, never
    #: rendered (T53).
    degenerate: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Names of PNGs in the accompanying ExplanationArtifacts.
    artifact_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe scalars only -- no arrays, no tensors, no Figures."""
        return {
            "label": self.label,
            "verdict": self.verdict,
            "p_fake": float(self.p_fake),
            "calibrated": bool(self.calibrated),
            "attribution": [a.to_dict() for a in self.attribution],
            "frequency": None if self.frequency is None else self.frequency.to_dict(),
            "timeline": [f.to_dict() for f in self.timeline],
            "spans": [s.to_dict() for s in self.spans],
            "degenerate": {k: bool(v) for k, v in self.degenerate.items()},
            "warnings": list(self.warnings),
            "artifact_names": list(self.artifact_names),
        }


@dataclass
class ExplanationArtifacts:
    """Rendered PNGs, keyed by name. Deliberately outside :class:`Explanation`.

    Kept separate so the JSON stays small (<10 KB) and streamable while the
    images are served as cacheable URLs (T58). A video's 16 CAM overlays would be
    ~1.6 MB of base64 inside the verdict payload otherwise.
    """

    images: dict[str, bytes] = field(default_factory=dict)

    def add(self, name: str, png: bytes) -> None:
        self.images[name] = png

    def names(self) -> list[str]:
        return sorted(self.images)


def decide_verdict(p_fake: float, margin: float = UNCERTAIN_MARGIN) -> str:
    """Map a fake probability to real | fake | uncertain.

    The `uncertain` band exists because a forced binary call on a near-0.5 margin
    is the most misleading output this system could produce -- especially while
    the model is uncalibrated, where the number near the boundary means even less
    than it appears to.
    """
    if abs(p_fake - 0.5) < margin:
        return "uncertain"
    return "fake" if p_fake >= 0.5 else "real"
