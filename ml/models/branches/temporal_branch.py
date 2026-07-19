"""Temporal branch for SEETHRU deepfake detection.

Operates on a sequence of per-frame spatial features (e.g. the 16 frame vectors
from :class:`SpatialBranch`) and models their temporal dynamics. Deepfakes often
exhibit subtle inter-frame inconsistencies (flicker, unstable identity/texture)
that a single frame cannot reveal.

Pipeline: ``(B, T, C)`` frame features → 2-layer bidirectional LSTM (hidden 256)
→ attention pooling over the ``T`` timesteps → ``(B, 512)`` feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LSTM_HIDDEN = 256
TEMPORAL_FEATURES = LSTM_HIDDEN * 2  # bidirectional -> 512


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention pooling over time.

    Learns a scalar relevance score per timestep and returns the weighted sum of
    the sequence — letting the branch focus on the most informative frames
    rather than averaging them equally.

    Args:
        feature_dim: Size of each timestep's feature vector.
        attn_dim: Hidden width of the scoring MLP.
    """

    def __init__(self, feature_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.project = nn.Linear(feature_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ``(B, T, F)`` to ``(B, F)``.

        Returns ``(pooled, weights)`` where ``weights`` is ``(B, T)`` and sums
        to 1 over the time dimension.
        """
        energy = torch.tanh(self.project(x))     # (B, T, attn_dim)
        scores = self.score(energy).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)   # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, F)
        return pooled, weights


class TemporalBranch(nn.Module):
    """BiLSTM + attention pooling over a frame-feature sequence → 512-dim vector.

    Args:
        input_size: Per-frame feature dimension ``C`` (e.g. 1536 from the
            spatial branch).
        hidden_size: LSTM hidden size per direction (default 256).
        num_layers: Number of stacked LSTM layers (default 2).
        dropout: Dropout between LSTM layers (applied when ``num_layers > 1``).
        attn_dim: Hidden width of the attention scoring MLP.
    """

    out_features: int = TEMPORAL_FEATURES

    def __init__(
        self,
        input_size: int = 1536,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int = 2,
        dropout: float = 0.2,
        attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size * 2, attn_dim=attn_dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        """Map ``(B, T, C)`` frame features to ``(B, 512)``.

        If ``return_attention`` is ``True``, also returns the ``(B, T)``
        attention weights (useful for explainability — which frames mattered).
        """
        # outputs: (B, T, 2*hidden) — hidden state at every timestep.
        outputs, _ = self.lstm(x)
        pooled, weights = self.attention(outputs)
        if return_attention:
            return pooled, weights
        return pooled


if __name__ == "__main__":
    model = TemporalBranch(input_size=1536)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("-" * 60)
    print("TemporalBranch (BiLSTM + attention)")
    print(f"  total parameters:   {n_params:,}")
    print(f"  output feature dim: {model.out_features}")

    # B=4 clips, T=16 frames, C=1536 spatial features per frame.
    dummy = torch.randn(4, 16, 1536)
    pooled, weights = model(dummy, return_attention=True)
    print(f"  input  shape:        {tuple(dummy.shape)}")
    print(f"  output shape:        {tuple(pooled.shape)}")
    print(f"  attention shape:     {tuple(weights.shape)}")
    print(f"  attention row sums:  {weights.sum(dim=1).tolist()}")  # ~1.0 each

    assert pooled.shape == (4, TEMPORAL_FEATURES), "unexpected output shape"
    assert weights.shape == (4, 16), "unexpected attention shape"
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)

    # Gradient check.
    dummy.requires_grad_(True)
    model(dummy).sum().backward()
    assert dummy.grad is not None and torch.isfinite(dummy.grad).all()
    print("  forward + backward OK")
