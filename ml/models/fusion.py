"""Feature fusion for SEETHRU deepfake detection.

Combines the per-branch features — spatial (1536), frequency (128), and
(optionally) temporal (512) — into a single 256-dim embedding that the
classification head consumes.

* :class:`FeatureFusion` — concatenate the branches and pass through an MLP.
* :class:`AttentionFusion` — learn a per-branch attention weight first, so the
  model can up-/down-weight whole branches per sample before concatenation.

Both return ``(B, 256)``. When temporal features are absent they are treated as
zeros, keeping the fused input fixed at 2176 dims.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

SPATIAL_DIM = 1536
FREQUENCY_DIM = 128
TEMPORAL_DIM = 512
FUSED_INPUT_DIM = SPATIAL_DIM + FREQUENCY_DIM + TEMPORAL_DIM  # 2176
FUSED_OUTPUT_DIM = 256


def _build_fusion_mlp(input_dim: int = FUSED_INPUT_DIM) -> nn.Sequential:
    """MLP [input -> 512 -> 256] with BatchNorm + ReLU + Dropout(0.4) per layer."""
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, FUSED_OUTPUT_DIM),
        nn.BatchNorm1d(FUSED_OUTPUT_DIM),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
    )


def _ensure_temporal(
    temporal: Optional[torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    """Return temporal features, or a matching zero tensor when absent."""
    if temporal is not None:
        return temporal
    return reference.new_zeros(reference.size(0), TEMPORAL_DIM)


class FeatureFusion(nn.Module):
    """Concatenate branch features and fuse them through an MLP → ``(B, 256)``."""

    out_features: int = FUSED_OUTPUT_DIM

    def __init__(self) -> None:
        super().__init__()
        self.mlp = _build_fusion_mlp(FUSED_INPUT_DIM)

    def forward(
        self,
        spatial: torch.Tensor,
        frequency: torch.Tensor,
        temporal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temporal = _ensure_temporal(temporal, spatial)
        fused = torch.cat([spatial, frequency, temporal], dim=1)  # (B, 2176)
        return self.mlp(fused)                                    # (B, 256)


class AttentionFusion(nn.Module):
    """Learn a per-branch attention weight, then fuse → ``(B, 256)``.

    Each branch is reduced to a scalar relevance score; a softmax across the
    three branches yields per-sample weights that scale each branch before
    concatenation. This lets the network rely more on, say, the frequency branch
    for one sample and the temporal branch for another.
    """

    out_features: int = FUSED_OUTPUT_DIM

    def __init__(self) -> None:
        super().__init__()
        # Per-branch scoring heads (feature vector -> scalar relevance).
        self.score_spatial = nn.Linear(SPATIAL_DIM, 1)
        self.score_frequency = nn.Linear(FREQUENCY_DIM, 1)
        self.score_temporal = nn.Linear(TEMPORAL_DIM, 1)
        self.mlp = _build_fusion_mlp(FUSED_INPUT_DIM)

    def forward(
        self,
        spatial: torch.Tensor,
        frequency: torch.Tensor,
        temporal: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        has_temporal = temporal is not None
        temporal = _ensure_temporal(temporal, spatial)

        # Per-branch relevance scores -> (B, 3).
        scores = torch.cat(
            [
                self.score_spatial(spatial),
                self.score_frequency(frequency),
                self.score_temporal(temporal),
            ],
            dim=1,
        )
        # Mask out the temporal branch entirely when it wasn't provided.
        if not has_temporal:
            scores = scores.clone()
            scores[:, 2] = float("-inf")

        weights = torch.softmax(scores, dim=1)  # (B, 3), rows sum to 1

        weighted = torch.cat(
            [
                spatial * weights[:, 0:1],
                frequency * weights[:, 1:2],
                temporal * weights[:, 2:3],
            ],
            dim=1,
        )  # (B, 2176)

        fused = self.mlp(weighted)  # (B, 256)
        if return_weights:
            return fused, weights
        return fused


if __name__ == "__main__":
    B = 4
    spatial = torch.randn(B, SPATIAL_DIM)
    frequency = torch.randn(B, FREQUENCY_DIM)
    temporal = torch.randn(B, TEMPORAL_DIM)

    for name, model in [
        ("FeatureFusion", FeatureFusion()),
        ("AttentionFusion", AttentionFusion()),
    ]:
        model.train()  # exercise BatchNorm/Dropout
        print("-" * 60)
        print(name, "| params:", f"{sum(p.numel() for p in model.parameters()):,}")

        # With temporal.
        out = model(spatial, frequency, temporal)
        print(f"  with temporal:    {tuple(out.shape)}")
        assert out.shape == (B, FUSED_OUTPUT_DIM)

        # Without temporal.
        out2 = model(spatial, frequency)
        print(f"  without temporal: {tuple(out2.shape)}")
        assert out2.shape == (B, FUSED_OUTPUT_DIM)

    # Inspect AttentionFusion branch weights.
    af = AttentionFusion().eval()
    _, w = af(spatial, frequency, temporal, return_weights=True)
    print("-" * 60)
    print("AttentionFusion branch weights (row sums ~1):")
    print("  weights[0]:", [round(v, 3) for v in w[0].tolist()])
    print("  row sums:  ", [round(v, 3) for v in w.sum(dim=1).tolist()])
    print("forward pass OK")
