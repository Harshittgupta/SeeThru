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

from typing import Dict

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
        self.spatial = SpatialBranch(pretrained=pretrained)
        self.frequency = FrequencyBranch()
        self.fusion = self._make_fusion(fusion)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(FUSED_DIM, num_classes),
        )

    @staticmethod
    def _make_fusion(fusion: str) -> nn.Module:
        if fusion == "concat":
            return FeatureFusion()
        if fusion == "attention":
            return AttentionFusion()
        raise ValueError(f"fusion must be 'concat' or 'attention', got {fusion!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, 224, 224)`` image batch → ``(B, num_classes)`` logits."""
        spatial = self.spatial(x)        # (B, 1536)
        frequency = self.frequency(x)    # (B, 128)
        fused = self.fusion(spatial, frequency, None)  # (B, 256)
        return self.classifier(fused)    # (B, num_classes)

    # ------------------------------------------------------------------ #
    def _ensure_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Add a batch dim if a single (unbatched) sample is passed."""
        if x.dim() == self._batched_ndim - 1:
            return x.unsqueeze(0)
        return x

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, object]:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, 3, 224, 224)`` clip → ``(B, num_classes)`` logits."""
        b, t = x.shape[:2]
        flat = x.reshape(b * t, *x.shape[2:])  # (B*T, 3, 224, 224)

        spatial_seq = self.spatial(flat).reshape(b, t, -1)      # (B, T, 1536)
        frequency_seq = self.frequency(flat).reshape(b, t, -1)  # (B, T, 128)

        # Pool spatial/frequency over time; temporal branch consumes the sequence.
        spatial = spatial_seq.mean(dim=1)        # (B, 1536)
        frequency = frequency_seq.mean(dim=1)    # (B, 128)
        temporal = self.temporal(spatial_seq)    # (B, 512)

        fused = self.fusion(spatial, frequency, temporal)  # (B, 256)
        return self.classifier(fused)


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
