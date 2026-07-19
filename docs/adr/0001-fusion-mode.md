# ADR 0001: Use concat fusion, and derive branch attribution from ablation

- **Status:** Accepted
- **Date:** 2026-07-15
- **Task:** BUILD_PLAN T20
- **Affects:** T25 (config default), T46/T51 (explainability contract), T58 (API response), T62 (frontend)

## Context

`DeepfakeClassifier` accepts `fusion="concat"` (`FeatureFusion`) or `fusion="attention"`
(`AttentionFusion`). The choice had to be made *before* training, because it decides what the
explainability layer is built against — and retraining to change it is expensive.

The apparent tension: **branch attribution** ("this call was driven by frequency evidence") is one of
this project's strongest explainability features, and `FeatureFusion` exposes no branch weights at
all. Only `AttentionFusion.forward(..., return_weights=True)` returns them. That seemed to force
`attention`, and an earlier draft of the plan recommended exactly that.

That recommendation was **wrong**. It rested on an assumption nobody had checked: that branch
attribution requires attention fusion.

## Evidence

Measured on 2026-07-15 (not reasoned from the source — run directly):

| Claim | Result |
|---|---|
| `score_temporal` gradient in stage 1 (image, temporal masked to `-inf`) | **norm exactly `0.0` — dead.** (`score_spatial`: 1038.7) |
| Stage 1 → stage 2 weight shift | A **per-sample uniform rescale** of the (spatial, frequency) block — identical factor for both branches (`allclose=True`), varying **0.41–0.89** across samples |
| Branch attribution from **concat** via leave-one-out ablation | **Works.** Per-sample, causal: zero a branch, measure Δ fake-logit |
| Cost of the attention gates | **+2,179 params (+0.175%)** — negligible |

## Decision

**Train with `fusion="concat"`. Derive branch attribution from leave-one-out ablation.**

This confirms the existing default in `classifier.py`, and matches the project spec's own
"Initial Recommendation: Feature Concatenation + MLP Classifier".

## Rationale

**1. Ablation is strictly better attribution than attention weights.**

|  | AttentionFusion weights | Ablation on concat |
|---|---|---|
| What it measures | a softmax over 3 learned scalars — a **gate** | a **causal** change in the actual output |
| Defensible phrasing | "what the model leaned on" | "removing frequency moves p_fake by −0.53" |
| "all three branches agree" | ❌ impossible — softmax forces sum = 1 | ✅ expressible |
| "no branch is confident" | ❌ impossible — same reason | ✅ expressible |
| Verifiable by a reader | no | yes — re-run it and check |

The zero-sum constraint is the deciding factor. Attention weights are *always relative*: they cannot
distinguish "every branch is confident" from "none of them are", because both normalise to the same
distribution. For a system whose entire thesis is honest explanation, an attribution that cannot
express agreement or uncertainty is a poor instrument. Ablation earns the stronger sentence honestly.

**2. The cost argument for ablation evaporates on inspection.** It needs three extra forward passes —
but only through *fusion + head*, because the branch features are already computed. The backbone,
which dominates cost, runs once regardless. Effectively free.

**3. Concat has the cleaner transfer path.** No dead head; no softmax re-normalisation between stages;
one fewer thing to go wrong in a two-stage schedule that is already delicate (T33).

**4. Attention's one real advantage stays unproven.** It offers genuine per-sample adaptivity that
concat structurally cannot replicate — a Linear applies identical weights to every sample, whereas the
gate is per-sample. On heavily compressed media, where frequency artifacts are destroyed, that *could*
help cross-dataset generalization (our weakest number, expected AUC 0.65–0.75). This is plausible and
worth testing. It is not worth *assuming*, and it does not justify accepting worse attribution to get
it.

## Consequences

- `ml/config.py` (T25) defaults to `fusion: "concat"`. `classifier.py` already does; no code change.
- **T51 changes**: branch attribution comes from an ablation helper in `ml/explainability/`, not from
  `AttentionFusion.return_weights`. It is now available on **every** checkpoint rather than only
  attention-trained ones — a simplification for T58 and T62.
- **T62 changes**: the attribution bar can use causal language ("removing this branch changes the score
  by X") instead of the hedged "what the model leaned on", and is no longer constrained to sum to 100%.
  The frontend must **not** render it as a pie chart or a normalised stacked bar — the values are
  independent Δs, not shares.
- Ablate to the **training-set mean** branch feature, not to zeros. Zeroing feeds the MLP a vector it
  never saw during training (off-manifold), which makes the measured Δ partly an artifact of the
  ablation itself. Compute the means once at the end of training and ship them in the checkpoint.
- `AttentionFusion` **stays in the codebase**, tested and working. It is one constructor argument away,
  and checkpoints record `fusion` (T31), so this decision is cheap to revisit.

## Revisit if

- Cross-dataset AUC lands at the low end (~0.65) and compression robustness (T77) shows the frequency
  branch collapsing on c40. That is the exact scenario per-sample gating is meant to fix, and would
  justify spending a second training run on the controlled A/B: identical seed and data, `concat` vs
  `attention`, compared on Celeb-DF's official 518-video subset.
- If that experiment runs, report it whatever the outcome. "We tried adaptive branch gating and it did
  not help" is a real finding, and a more honest one than quietly keeping only the better number.
