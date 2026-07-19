"""Branch attribution by ablation (BUILD_PLAN T51, docs/adr/0001-fusion-mode.md).

Answers "which branch drove this call?" by removing each branch and measuring
what changes.

**Why ablation and not AttentionFusion's weights.** ADR 0001 decided this after
measuring. In short:

  * Attention weights are a **softmax over three learned scalars** -- a gate
    score. Ablation is a **causal** measurement: "removing frequency moves the
    fake logit by -0.53" is checkable; you can re-run it.
  * The softmax forces the three weights to sum to 1, so attention **cannot
    express** "all three branches agree strongly" or "none of them are driving
    this". Both are informative, and ablation reports both naturally.
  * Ablation works on **any** checkpoint, so the backend needs no `isinstance`
    branching and no "degrade to None on concat" path.

**Ablate to the training-set MEAN, not to zero.** Zero is off-manifold: the
fusion MLP never saw a zero vector in training, so part of the measured delta
would be the network reacting to an impossible input rather than to the branch's
absence. The means are computed once at the end of training and shipped in the
checkpoint (T24's ``branch_means``).

**Cost is ~free.** ``fuse_and_classify`` re-runs only fusion+head on cached
branch features, so three ablations cost three tiny MLP passes -- the backbone,
which dominates, runs once.
"""

from __future__ import annotations

import logging

import torch

from ml.explainability.contracts import BranchAttribution

logger = logging.getLogger(__name__)

# Deltas below this are reported but flagged: the branch is not moving the
# decision, and the UI should not imply otherwise.
NEGLIGIBLE_DELTA = 1e-3


def compute_branch_means(model, loader, device, max_batches: int = 50) -> dict:
    """Mean branch features over (a sample of) the training set.

    Run once at the end of training; store in the checkpoint. 50 batches is
    plenty -- this is a mean over a 1536-dim vector, not a distribution estimate,
    and it converges quickly.
    """
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    n = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = (batch["frames"] if isinstance(batch, dict) else batch[0]).to(device)
            _logits, aux = model.forward_explain(x)
            for branch in ("spatial", "frequency", "temporal"):
                value = aux.get(branch)
                if value is None:
                    continue
                total = value.detach().float().sum(dim=0).cpu()
                sums[branch] = sums[branch] + total if branch in sums else total
            n += x.shape[0]

    if not n:
        raise ValueError("no batches -- cannot compute branch means")
    return {k: (v / n).tolist() for k, v in sums.items()}


def _baseline_for(
    branch: str, value: torch.Tensor, branch_means: dict | None
) -> tuple[torch.Tensor, str]:
    """The tensor to replace ``value`` with, and what to call it."""
    if branch_means and branch in branch_means:
        mean = torch.tensor(
            branch_means[branch], dtype=value.dtype, device=value.device
        )
        return mean.unsqueeze(0).expand_as(value), "mean"

    # Zeros are off-manifold; say so rather than quietly reporting a number that
    # is partly an artifact of the ablation itself.
    logger.warning(
        "No training-set mean for the %r branch, falling back to zeros. Zeros are "
        "off-manifold (the fusion MLP never saw one in training), so part of the "
        "measured delta is the network reacting to an impossible input. Compute "
        "branch_means at the end of training and ship them in the checkpoint (T24).",
        branch,
    )
    return torch.zeros_like(value), "zero"


@torch.no_grad()
def branch_attribution(
    model,
    aux: dict,
    branch_means: dict | None = None,
    target_class: int = 1,
) -> list[BranchAttribution]:
    """Leave-one-out attribution over the fused branches.

    Args:
        model: A ``DeepfakeClassifier`` (needs ``fuse_and_classify``).
        aux: The dict from ``model.forward_explain(x)`` -- branch features,
            already computed, so the backbone does not run again.
        branch_means: ``checkpoint["branch_means"]``. Strongly recommended.
        target_class: 1 = fake.

    Returns one entry per branch **actually present**. For an image model there
    is no temporal branch, so it emits **2 entries, not 3 with a zero** -- "0%
    temporal" reads as a measurement when it is a structural absence (T51).
    """
    spatial, frequency, temporal = aux["spatial"], aux["frequency"], aux.get("temporal")

    baseline_logits = model.fuse_and_classify(spatial, frequency, temporal)
    baseline = baseline_logits[:, target_class]

    present = {"spatial": spatial, "frequency": frequency}
    if temporal is not None:
        present["temporal"] = temporal

    out: list[BranchAttribution] = []
    for branch, value in present.items():
        ablated_value, baseline_kind = _baseline_for(branch, value, branch_means)
        args = {"spatial": spatial, "frequency": frequency, "temporal": temporal}
        args[branch] = ablated_value

        ablated = model.fuse_and_classify(
            args["spatial"], args["frequency"], args["temporal"]
        )[:, target_class]

        delta = float((baseline - ablated).mean().item())
        out.append(BranchAttribution(branch=branch, delta=delta, baseline=baseline_kind))

    return out


def describe(attribution: list[BranchAttribution]) -> list[str]:
    """Human-readable sentences for the UI.

    Phrased causally because ablation earns it: "removing X changes the score by
    Y" is a re-runnable fact, unlike a gate weight. But the wording stays
    relative -- the model is uncalibrated (T78), so the *magnitude* of a logit
    delta is not something to quote as evidence strength.
    """
    if not attribution:
        return []

    ranked = sorted(attribution, key=lambda a: abs(a.delta), reverse=True)
    lines: list[str] = []

    strongest = ranked[0]
    if abs(strongest.delta) < NEGLIGIBLE_DELTA:
        lines.append(
            "No single branch is driving this prediction -- removing any one of "
            "them barely changes the score."
        )
        return lines

    lines.append(
        f"Removing the {strongest.branch} branch changes the fake score the most "
        f"({strongest.delta:+.3f} logits), so it contributed most to this call."
    )
    agreeing = [a for a in ranked if a.delta > NEGLIGIBLE_DELTA]
    if len(agreeing) == len(ranked) and len(ranked) > 1:
        # A statement AttentionFusion's softmax literally cannot make.
        lines.append(
            f"All {len(ranked)} branches push in the same direction -- they agree."
        )
    return lines
