"""Frequency-domain evidence (BUILD_PLAN T49).

The frequency branch is the least intuitive part of this model, so its
explanation has to do the most work. Three artifacts, in descending order of how
much they actually convince a non-expert:

1. **High-frequency energy ratio** -- one number, one sentence. *"This face has
   87% more high-frequency energy than a typical real face."* Ship this first; it
   is the part someone can act on.
2. **Radial power profile** -- the azimuthally-averaged spectrum, plotted against
   real/fake reference bands. GAN upsampling leaves a periodic signature that
   shows up as a bump in the high-frequency tail. **The reference curves are not
   optional decoration**: a lone profile is an unlabelled squiggle, and a viewer
   has no way to know whether its tail is unusual. Compute the reference means
   once over the training set and ship them alongside the weights.
3. **The spectrum heatmap** -- pretty, and essentially unreadable to a
   non-expert. Keep it as the small panel, not the headline.

**No GradCAM over this branch.** Its H/W axes are FFT coordinates, not image
coordinates, so a CAM drawn over them cannot be overlaid on a face without
asserting something false about *where* in the picture the evidence is. The
radial profile says the same thing legibly.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ml.explainability.contracts import FrequencyEvidence

logger = logging.getLogger(__name__)

#: Radius (as a fraction of Nyquist) above which energy counts as "high
#: frequency". 0.5 = the half-Nyquist ring; upsampling artifacts live above it.
HIGH_FREQ_RADIUS = 0.5
N_RADIAL_BINS = 64


def _radius_map(h: int, w: int) -> np.ndarray:
    """Normalized distance from the (centred) DC term, in [0, ~1]."""
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return r / r.max()


def log_spectrum(image: torch.Tensor) -> np.ndarray:
    """Centred log-magnitude spectrum of one image → ``(H, W)``, channel-averaged.

    Reuses the model's own :func:`log_magnitude_spectrum`, deliberately: an
    explanation computed with a *different* transform than the branch uses would
    be describing a spectrum the model never saw.
    """
    from ml.models.branches.frequency_branch import log_magnitude_spectrum

    if image.dim() == 3:
        image = image.unsqueeze(0)
    with torch.no_grad():
        spec = log_magnitude_spectrum(image.float())
    return spec[0].mean(dim=0).cpu().numpy()


def high_frequency_ratio(spectrum: np.ndarray, radius: float = HIGH_FREQ_RADIUS) -> float:
    """Share of spectral energy above ``radius`` × Nyquist.

    The log-spectrum is shifted to be non-negative before summing: log magnitudes
    are frequently negative, and a "ratio" of signed quantities is not a ratio --
    it can exceed 1 or flip sign, and would be quietly nonsensical.
    """
    r = _radius_map(*spectrum.shape)
    shifted = spectrum - spectrum.min()
    total = shifted.sum()
    if total <= 0:
        return 0.0
    return float(shifted[r >= radius].sum() / total)


def radial_profile(spectrum: np.ndarray, n_bins: int = N_RADIAL_BINS) -> np.ndarray:
    """Azimuthally-averaged log magnitude, DC → Nyquist → ``(n_bins,)``.

    Averaging over angle collapses the 2D spectrum into the one axis that carries
    the artifact: GAN/diffusion upsampling is (approximately) isotropic, so its
    signature is a function of frequency magnitude, not direction.
    """
    r = _radius_map(*spectrum.shape)
    bins = np.linspace(0, r.max(), n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        out[i] = spectrum[mask].mean() if mask.any() else (out[i - 1] if i else 0.0)
    return out


def frequency_evidence(
    image: torch.Tensor, references: dict | None = None
) -> tuple[FrequencyEvidence, np.ndarray]:
    """→ ``(FrequencyEvidence, spectrum)`` for one image.

    Args:
        image: A normalized ``(3, H, W)`` or ``(1, 3, H, W)`` model input.
        references: ``{"real": [...], "fake": [...]}`` radial profiles averaged
            over the training set. Without them the profile is unreadable -- see
            the module docstring.
    """
    spectrum = log_spectrum(image)
    profile = radial_profile(spectrum)
    ratio = high_frequency_ratio(spectrum)

    references = references or {}
    if not references:
        logger.debug(
            "No radial reference profiles supplied; the UI will render this "
            "profile without real/fake bands, which makes it hard to read."
        )

    return (
        FrequencyEvidence(
            hf_energy_ratio=ratio,
            radial_profile=profile.tolist(),
            reference_real=list(references.get("real", [])),
            reference_fake=list(references.get("fake", [])),
        ),
        spectrum,
    )


def describe(evidence: FrequencyEvidence) -> list[str]:
    """One plain sentence about the frequency evidence.

    Phrased *relative to the reference* when we have one, and hedged when we do
    not: an absolute high-frequency ratio means nothing without knowing what a
    normal face looks like, and quoting one as if it did would be false
    precision.
    """
    ratio = evidence.hf_energy_ratio
    if not evidence.reference_real:
        return [
            f"High-frequency energy: {ratio:.1%} of the spectrum. "
            f"(No reference profile available, so this cannot be compared to a "
            f"typical real face.)"
        ]

    reference = np.asarray(evidence.reference_real, dtype=float)
    shifted = reference - reference.min()
    total = shifted.sum()
    if total <= 0:
        return [f"High-frequency energy: {ratio:.1%} of the spectrum."]

    baseline = float(shifted[len(shifted) // 2 :].sum() / total)
    if baseline <= 0:
        return [f"High-frequency energy: {ratio:.1%} of the spectrum."]

    relative = (ratio - baseline) / baseline
    if abs(relative) < 0.05:
        return ["High-frequency energy is typical of a real face."]
    direction = "more" if relative > 0 else "less"
    return [
        f"This face has {abs(relative):.0%} {direction} high-frequency energy than "
        f"a typical real face -- the band where upsampling artifacts appear."
    ]


def compute_reference_profiles(loader, device, max_batches: int = 50) -> dict:
    """Mean radial profile per class over the training set.

    Run once after training; ship next to the weights. Cheap (an FFT per image,
    no model) and it is what turns the profile plot from a squiggle into evidence.
    """
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y = (batch["frames"], batch["label"]) if isinstance(batch, dict) else batch
        for image, label in zip(x, y, strict=True):
            profile = radial_profile(log_spectrum(image.to(device)))
            key = int(label)
            sums[key] = sums.get(key, np.zeros_like(profile)) + profile
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        raise ValueError("no batches -- cannot compute reference profiles")
    return {
        name: (sums[key] / counts[key]).tolist()
        for key, name in ((0, "real"), (1, "fake"))
        if key in counts
    }
