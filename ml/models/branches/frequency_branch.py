"""Frequency branch for SEETHRU deepfake detection.

Many generative models leave tell-tale periodic artifacts (upsampling /
GAN-fingerprint patterns) that are far more visible in the Fourier domain than
in pixel space. This branch transforms each image into its log-magnitude
spectrum and learns a small CNN over it, producing a 128-dim feature vector.

Pipeline: ``(B, 3, 224, 224)`` image → 2D FFT (differentiable, on-device) →
log-magnitude spectrum → 3-layer CNN (32→64→128, each Conv+BN+ReLU+MaxPool) →
global average pool → ``(B, 128)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FREQUENCY_FEATURES = 128


def log_magnitude_spectrum(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Centered log-magnitude Fourier spectrum of an image batch.

    Args:
        x: ``(B, C, H, W)`` image tensor (any device; the transform runs on it).
        eps: Stabilizer so ``log`` stays finite at zero magnitude.

    Returns:
        ``(B, C, H, W)`` real tensor. Fully differentiable — ``torch.fft.fft2``
        and the elementwise ops all support autograd, and everything stays on
        the input's device (CPU or GPU).
    """
    # 2D FFT over the spatial dims; low frequencies shifted to the centre.
    freq = torch.fft.fft2(x, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    magnitude = torch.abs(freq)
    return torch.log(magnitude + eps)


class FrequencyBranch(nn.Module):
    """CNN over the log-magnitude FFT spectrum → 128-dim feature vector."""

    out_features: int = FREQUENCY_FEATURES

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, 224, 224)`` images to ``(B, 128)`` frequency features."""
        spectrum = log_magnitude_spectrum(x)  # differentiable, on-device
        feats = self.features(spectrum)        # (B, 128, 28, 28)
        feats = self.pool(feats)               # (B, 128, 1, 1)
        return torch.flatten(feats, 1)         # (B, 128)


def visualize_spectrum(
    image_tensor: torch.Tensor, save_path: str = "spectrum.png"
) -> str:
    """Save the log-magnitude spectrum of one image as a matplotlib figure.

    Accepts a ``(C, H, W)`` or ``(B, C, H, W)`` tensor (the first sample of a
    batch is used). Channels are averaged into a single grayscale spectrum.
    Returns the path written.
    """
    import matplotlib.pyplot as plt

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        spectrum = log_magnitude_spectrum(image_tensor.float())

    # First sample, mean over channels -> (H, W), to CPU for plotting.
    spec = spectrum[0].mean(dim=0).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(spec, cmap="viridis")
    ax.set_title("Log-magnitude FFT spectrum")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    model = FrequencyBranch()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("-" * 60)
    print("FrequencyBranch")
    print(f"  total parameters:   {n_params:,}")
    print(f"  output feature dim: {model.out_features}")

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"  input  shape: {tuple(dummy.shape)}")
    print(f"  output shape: {tuple(out.shape)}")
    assert out.shape == (2, FREQUENCY_FEATURES), "unexpected output shape"

    # Differentiability check: gradient must flow back through the FFT.
    dummy.requires_grad_(True)
    model(dummy).sum().backward()
    assert dummy.grad is not None and torch.isfinite(dummy.grad).all()
    print("  forward + backward through FFT OK")

    path = visualize_spectrum(dummy.detach()[0])
    print(f"  saved spectrum -> {path}")
