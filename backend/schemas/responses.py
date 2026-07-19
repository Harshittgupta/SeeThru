"""Response schemas (BUILD_PLAN T58).

The response contract, and the honesty rules baked into it:

* **Artifacts are URLs, not base64.** A video result references 16 heatmaps + a
  spectrum + a timeline; base64-ing them is ~1.6 MB of text in the verdict
  payload that must be fully buffered and parsed before anything shows, and
  cannot be cached. Measured in Milestone 5: one spectrum PNG alone is 304 KB.
  So the JSON carries relative URLs and the PNGs stream from /v1/artifacts.
* **verdict is real | fake | uncertain.** A forced binary call on a near-0.5
  margin is the single most misleading thing this API could return.
* **calibrated says whether the score is a probability.** It is False today (no
  calibration exists, T78), and the frontend keys off it to suppress the
  percentage (T63).
* **disclaimer is non-strippable, on every prediction.** Detectors trained on
  FF++ generalise poorly; the caller is told in-band, not in docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str = "ok"
    version: str
    uptime_s: float


class Readiness(BaseModel):
    ready: bool
    reason: str | None = None


class BranchAttributionOut(BaseModel):
    branch: str
    delta: float = Field(description="Causal change in the fake logit when this branch is ablated. NOT a share; the branches do not sum to 1.")
    baseline: str


class FaceResult(BaseModel):
    face_id: int
    bbox: list[int] = Field(description="[x1, y1, x2, y2] in ORIGINAL image pixels.")
    verdict: str = Field(description="real | fake | uncertain")
    scores: dict[str, float]
    attribution: list[BranchAttributionOut] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="name -> relative URL under /v1/artifacts. Never base64.",
    )
    hf_energy_ratio: float | None = None


class Disclaimer(BaseModel):
    not_forensic_evidence: bool = True
    known_limitations: list[str] = Field(default_factory=list)


class ModelBlock(BaseModel):
    arch: str
    version: str | None = None
    device: str
    calibrated: bool = Field(description="If false, scores are NOT probabilities. Do not render as a percentage.")


class ImagePrediction(BaseModel):
    request_id: str | None = None
    model: ModelBlock
    media: dict
    faces: list[FaceResult]
    summary: dict = Field(description="verdict, confidence, reasoning[], any_fake")
    warnings: list[str] = Field(default_factory=list)
    disclaimer: Disclaimer


class FrameScoreOut(BaseModel):
    index: int
    source_index: int
    t_seconds: float | None = None
    p_fake: float
    attention_norm: float | None = None
    suspicious: bool = False
    interpolated: bool = Field(False, description="Face copied from a neighbour; NOT an observation.")


class SpanOut(BaseModel):
    start_s: float
    end_s: float
    mean_p_fake: float
    n_frames: int


class VideoResult(BaseModel):
    request_id: str | None = None
    model: ModelBlock
    media: dict
    verdict: str
    scores: dict[str, float]
    attribution: list[BranchAttributionOut] = Field(default_factory=list)
    timeline: list[FrameScoreOut] = Field(default_factory=list)
    spans: list[SpanOut] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: Disclaimer


class JobAccepted(BaseModel):
    job_id: str
    state: str = "queued"
    poll_url: str


class JobStatus(BaseModel):
    job_id: str
    state: str = Field(description="queued | running | succeeded | failed | expired")
    progress: float = 0.0
    stage: str | None = None
    error_code: str | None = None
    created_at: float
    poll_url: str
