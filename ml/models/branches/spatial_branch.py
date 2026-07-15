"""Spatial branch for SEETHRU deepfake detection.

Wraps a torchvision EfficientNet-B3 as a feature extractor: the ImageNet
classification head is removed and replaced with global average pooling, so the
branch maps a ``(B, 3, 224, 224)`` image batch to a ``(B, 1536)`` feature
vector (1536 = EfficientNet-B3's final-stage channel width).

These spatial features capture per-frame appearance cues (blending boundaries,
texture/warping artifacts) and are later fused with the other detection
branches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

# EfficientNet-B3's last convolutional stage outputs 1536 channels.
EFFICIENTNET_B3_FEATURES = 1536


class SpatialBranch(nn.Module):
    """EfficientNet-B3 backbone producing a 1536-dim feature vector.

    Args:
        input_channels: Number of input image channels (default 3). When not 3,
            the stem convolution is replaced to accept the new channel count.
        pretrained: If ``True``, load ImageNet-pretrained weights.
    """

    out_features: int = EFFICIENTNET_B3_FEATURES

    def __init__(self, input_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()

        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b3(weights=weights)

        # Drop the classifier head; keep the convolutional feature extractor.
        self.features = backbone.features

        # Replace the stem conv if the input channel count differs from 3.
        if input_channels != 3:
            self._adapt_input_channels(input_channels)

        # Global average pooling -> one value per channel -> (B, 1536).
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _adapt_input_channels(self, input_channels: int) -> None:
        """Swap the stem convolution to accept ``input_channels`` inputs."""
        stem_conv = self.features[0][0]  # Conv2dNormActivation -> Conv2d
        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=stem_conv.bias is not None,
        )
        self.features[0][0] = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, 224, 224)`` to ``(B, 1536)`` features."""
        x = self.features(x)          # (B, 1536, 7, 7)
        x = self.pool(x)              # (B, 1536, 1, 1)
        x = torch.flatten(x, 1)       # (B, 1536)
        return x


if __name__ == "__main__":
    model = SpatialBranch(input_channels=3, pretrained=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print("-" * 60)
    print(f"SpatialBranch (EfficientNet-B3)")
    print(f"  total parameters:     {n_params:,}")
    print(f"  trainable parameters: {trainable:,}")
    print(f"  output feature dim:   {model.out_features}")

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"  input  shape: {tuple(dummy.shape)}")
    print(f"  output shape: {tuple(out.shape)}")
    assert out.shape == (2, EFFICIENTNET_B3_FEATURES), "unexpected output shape"
    print("  forward pass OK")
