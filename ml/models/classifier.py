"""Deepfake classifier heads for SEETHRU.

Composes the feature branches + fusion + a linear head into end-to-end
classifiers that output real/fake logits.

* :class:`DeepfakeClassifier` — base: spatial + frequency branches → fusion →
  linear head. Operates on single images.
* :class:`ImageClassifier` — image model (no temporal branch).
* :class:`VideoClassifier` — adds the temporal (BiLSTM) branch over a 16-frame
  clip; spatial/frequency features are pooled over time for fusion.

All produce ``(B, 2)`` logits and share a :meth:`predict` helper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # package import
    from .branches import FrequencyBranch, SpatialBranch, TemporalBranch
    from .fusion import AttentionFusion, FeatureFusion
except ImportError:  # pragma: no cover - direct-script execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from branches import FrequencyBranch, SpatialBranch, TemporalBranch
    from fusion import AttentionFusion, FeatureFusion

CLASS_NAMES = ("real", "fake")
FUSED_DIM = 256


class DeepfakeClassifier(nn.Module):
    """Spatial + frequency branches → fusion → linear head (image-level).

    Args:
        num_classes: Number of output classes (default 2: real/fake).
        dropout: Dropout applied before the final linear layer.
        pretrained: Load ImageNet weights for the spatial backbone.
        fusion: ``"concat"`` (:class:`FeatureFusion`) or ``"attention"``
            (:class:`AttentionFusion`).
    """

    # Rank of a *batched* input tensor (image: (B, C, H, W) = 4).
    _batched_ndim = 4

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.4,
        pretrained: bool = True,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.fusion_mode = fusion
        self.spatial = SpatialBranch(pretrained=pretrained)
        self.frequency = FrequencyBranch()
        # dropout is threaded into fusion too (T22). It previously reached only
        # the final head below, while the fusion MLP stayed hardcoded at 0.4 --
        # so the spec's "dropout 0.3-0.5" was unreachable from config, and the
        # 1.2M-parameter fusion MLP (where most of the overfitting risk lives)
        # ignored the knob entirely.
        self.fusion = self._make_fusion(fusion, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(FUSED_DIM, num_classes),
        )

    @staticmethod
    def _make_fusion(fusion: str, dropout: float = 0.4) -> nn.Module:
        if fusion == "concat":
            return FeatureFusion(dropout=dropout)
        if fusion == "attention":
            return AttentionFusion(dropout=dropout)
        raise ValueError(f"fusion must be 'concat' or 'attention', got {fusion!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, 224, 224)`` image batch → ``(B, num_classes)`` logits."""
        logits, _aux = self._forward_impl(x, collect_aux=False)
        return logits

    def forward_explain(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        """Like :meth:`forward`, but also return the intermediates explainability needs.

        Returns ``(logits, aux)``. ``aux`` carries the per-branch feature vectors,
        which is what makes ablation attribution cheap (T51): the backbone runs
        once here, and :meth:`fuse_and_classify` then re-runs only fusion+head per
        ablated branch.

        aux keys:
            spatial      (B, 1536)      pooled spatial features
            frequency    (B, 128)       pooled frequency features
            temporal     (B, 512)|None  None for images (no temporal branch)
            branch_weights (B, 3)|None  AttentionFusion gates, if that fusion is
                                        in use. None under concat -- which is the
                                        default (ADR 0001), because attribution
                                        comes from ablation instead.

        This exists because forward() previously hardcoded
        ``self.fusion(spatial, frequency, None)`` and dropped everything, so the
        intermediates were unreachable from outside (T23).
        """
        return self._forward_impl(x, collect_aux=True)

    def _forward_impl(
        self, x: torch.Tensor, collect_aux: bool = False
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        """Single implementation behind forward() and forward_explain().

        One code path on purpose: if the explain path ran different code from the
        training path, the explanation would describe a model that never ran.
        """
        spatial = self.spatial(x)        # (B, 1536)
        frequency = self.frequency(x)    # (B, 128)

        fused, weights = self._fuse(spatial, frequency, None, want_weights=collect_aux)
        logits = self.classifier(fused)  # (B, num_classes)

        if not collect_aux:
            return logits, None
        return logits, {
            "spatial": spatial,
            "frequency": frequency,
            "temporal": None,
            "branch_weights": weights,
        }

    def _fuse(
        self,
        spatial: torch.Tensor,
        frequency: torch.Tensor,
        temporal: torch.Tensor | None,
        want_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse, optionally also returning AttentionFusion's branch weights.

        Only AttentionFusion accepts return_weights, so this isolates the one
        place that has to care which fusion is configured.
        """
        if want_weights and isinstance(self.fusion, AttentionFusion):
            fused, weights = self.fusion(
                spatial, frequency, temporal, return_weights=True
            )
            return fused, weights
        return self.fusion(spatial, frequency, temporal), None

    def fuse_and_classify(
        self,
        spatial: torch.Tensor,
        frequency: torch.Tensor,
        temporal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run **only** fusion + head on pre-computed branch features → logits.

        The workhorse of ablation attribution (T51). Branch features come from
        :meth:`forward_explain`; this then re-runs the cheap tail with one branch
        replaced by its training-set mean, and the logit delta is that branch's
        causal contribution. Because the backbone -- which dominates cost -- is
        skipped, a 3-branch attribution costs ~3 tiny MLP passes rather than 3
        full forwards.
        """
        return self.classifier(self.fusion(spatial, frequency, temporal))

    # ------------------------------------------------------------------ #
    def _ensure_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Add a batch dim if a single (unbatched) sample is passed."""
        if x.dim() == self._batched_ndim - 1:
            return x.unsqueeze(0)
        return x

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> dict[str, object]:
        """Classify a single sample → ``{label, confidence, logits}``.

        Accepts an unbatched sample or a batch of one. For multi-sample batches
        use :meth:`forward` directly; this reports on the first sample.
        """
        was_training = self.training
        self.eval()
        x = self._ensure_batch(image_tensor)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, idx = probs[0].max(dim=0)
        if was_training:
            self.train()
        return {
            "label": CLASS_NAMES[int(idx)],
            "confidence": float(confidence),
            "logits": logits[0].detach().cpu(),
        }


class ImageClassifier(DeepfakeClassifier):
    """Single-image deepfake classifier (spatial + frequency, no temporal)."""

    # Inherits the image forward/predict from DeepfakeClassifier unchanged.


class VideoClassifier(DeepfakeClassifier):
    """Clip-level classifier: adds a temporal BiLSTM over the frame sequence.

    Input is a ``(B, T, 3, 224, 224)`` clip. The spatial branch runs on every
    frame; the resulting ``(B, T, 1536)`` sequence drives the temporal branch,
    while spatial and frequency features are mean-pooled over time for fusion.
    """

    # Batched clip is (B, T, C, H, W) = 5 dims.
    _batched_ndim = 5

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.4,
        pretrained: bool = True,
        fusion: str = "concat",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            fusion=fusion,
        )
        self.temporal = TemporalBranch(input_size=self.spatial.out_features)

    def _forward_impl(
        self, x: torch.Tensor, collect_aux: bool = False
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        """``(B, T, 3, 224, 224)`` clip → ``(B, num_classes)`` logits (+ aux)."""
        b, t = x.shape[:2]
        flat = x.reshape(b * t, *x.shape[2:])  # (B*T, 3, 224, 224)

        spatial_seq = self.spatial(flat).reshape(b, t, -1)      # (B, T, 1536)
        frequency_seq = self.frequency(flat).reshape(b, t, -1)  # (B, T, 128)

        # Pool spatial/frequency over time; temporal branch consumes the sequence.
        spatial = spatial_seq.mean(dim=1)        # (B, 1536)
        frequency = frequency_seq.mean(dim=1)    # (B, 128)

        if collect_aux:
            temporal, attn = self.temporal(spatial_seq, return_attention=True)
        else:
            temporal, attn = self.temporal(spatial_seq), None  # (B, 512)

        fused, weights = self._fuse(
            spatial, frequency, temporal, want_weights=collect_aux
        )
        logits = self.classifier(fused)

        if not collect_aux:
            return logits, None
        return logits, {
            "spatial": spatial,
            "frequency": frequency,
            "temporal": temporal,
            "branch_weights": weights,
            # Per-frame sequences. T50 reuses these to score each frame
            # individually (fusion+head with temporal=None) for the manipulation
            # timeline -- for free, since they are already computed here.
            "spatial_seq": spatial_seq,      # (B, T, 1536)
            "frequency_seq": frequency_seq,  # (B, T, 128)
            # (B, T) softmax over time. NOTE: rows sum to 1, so at T=16 uniform
            # is 0.0625 and the spec's raw 0.6 threshold can never fire -- T50
            # normalizes by the row max instead.
            "temporal_attn": attn,
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("ImageClassifier")
    img_model = ImageClassifier(pretrained=False)
    img_model.eval()
    imgs = torch.randn(2, 3, 224, 224)
    logits = img_model(imgs)
    print(f"  input {tuple(imgs.shape)} -> logits {tuple(logits.shape)}")
    assert logits.shape == (2, 2)
    pred = img_model.predict(imgs[0])
    print(f"  predict: label={pred['label']} "
          f"confidence={pred['confidence']:.3f} logits={pred['logits'].tolist()}")

    print("=" * 60)
    print("VideoClassifier")
    vid_model = VideoClassifier(pretrained=False)
    vid_model.eval()
    clips = torch.randn(2, 16, 3, 224, 224)
    logits = vid_model(clips)
    print(f"  input {tuple(clips.shape)} -> logits {tuple(logits.shape)}")
    assert logits.shape == (2, 2)
    pred = vid_model.predict(clips[0])
    print(f"  predict: label={pred['label']} "
          f"confidence={pred['confidence']:.3f} logits={pred['logits'].tolist()}")
    print("=" * 60)
    print("forward + predict OK")
