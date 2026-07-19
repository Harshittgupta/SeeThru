"""Explainability for SEETHRU (BUILD_PLAN Milestone 5).

The "SEE" in SEETHRU. One entry point::

    from ml.explainability import Explainer

    explainer = Explainer(model, meta)          # meta from checkpoint.load_for_inference
    explanation, artifacts = explainer.explain_image(x)
    payload = explanation.to_dict()             # JSON-safe; images live in `artifacts`

Design decisions worth knowing before editing anything here:

* **Branch attribution comes from ablation, not attention weights**
  (docs/adr/0001-fusion-mode.md). Ablation is causal and re-runnable; a softmax
  gate is neither, and cannot express "all branches agree".
* **GradCAM is hand-rolled** (T47). The `grad-cam` library misreads a 5D clip as
  a 3D conv volume and assumes activation-batch == input-batch, which
  VideoClassifier's (B,T,...)->(B*T,...) flatten violates.
* **The spec's raw 0.6 attention threshold cannot fire** (T50). Attention is a
  softmax over T; the measured max over 200 clips was 0.0662. We threshold on
  normalized attention instead.
* **A degenerate heatmap is reported, never rendered** (T53). Min-max normalising
  a dead CAM turns float noise into a confident-looking map.
* **`matplotlib.use("Agg")` in render.py must run before any pyplot import** --
  a GUI backend inside a FastAPI worker crashes or leaks figures.
"""

from ml.explainability.contracts import (
    BranchAttribution,
    Explanation,
    ExplanationArtifacts,
    FrameScore,
    FrequencyEvidence,
    TimelineSpan,
    decide_verdict,
)
from ml.explainability.explainer import Explainer
from ml.explainability.gradcam import GradCAM, is_degenerate, spatial_target_layer

__all__ = [
    "BranchAttribution",
    "Explainer",
    "Explanation",
    "ExplanationArtifacts",
    "FrameScore",
    "FrequencyEvidence",
    "GradCAM",
    "TimelineSpan",
    "decide_verdict",
    "is_degenerate",
    "spatial_target_layer",
]
