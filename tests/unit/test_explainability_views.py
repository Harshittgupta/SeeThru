"""Frequency + temporal explainability (BUILD_PLAN T49/T50).

The load-bearing test is `test_raw_softmax_attention_can_never_reach_the_spec_threshold`.
The spec asks for a 0.6 attention threshold; attention is a softmax over 16
frames, so uniform is 0.0625 and the max seen over 200 random clips was 0.0662.
The threshold flags zero frames, forever, and looks like "nothing detected".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.explainability.frequency_viz import describe as describe_frequency
from ml.explainability.frequency_viz import (
    frequency_evidence,
    high_frequency_ratio,
    log_spectrum,
    radial_profile,
)
from ml.explainability.temporal_viz import (
    ATTENTION_THRESHOLD,
    attention_relative_to_uniform,
    build_timeline,
    merge_spans,
    normalize_attention,
    per_frame_scores,
)
from ml.explainability.temporal_viz import describe as describe_timeline


# --------------------------------------------------------------------------- #
# T50: the threshold that cannot fire
# --------------------------------------------------------------------------- #
def test_raw_softmax_attention_can_never_reach_the_spec_threshold():
    """Measured, not argued.

    The spec says "Attention Threshold: 0.6". TemporalAttention softmaxes over
    T, so 16 weights sum to 1 and uniform is 0.0625. A raw >=0.6 test flags zero
    frames -- and reports that as "no manipulation", not "unreachable threshold".
    """
    from ml.models.branches import TemporalBranch

    torch.manual_seed(0)
    branch = TemporalBranch(input_size=1536).eval()

    peaks = []
    with torch.no_grad():
        for _ in range(30):
            _, w = branch(torch.randn(1, 16, 1536), return_attention=True)
            assert w.sum().item() == pytest.approx(1.0, abs=1e-5)
            peaks.append(w.max().item())

    assert max(peaks) < 0.6, "premise broken -- re-check T50's reasoning"
    assert max(peaks) < 0.2, (
        f"the largest attention weight over 30 clips was {max(peaks):.4f}; a raw "
        f"0.6 threshold is unreachable, which is why we normalize by the peak"
    )


def test_normalized_attention_makes_the_threshold_meaningful():
    raw = np.array([0.02, 0.05, 0.10, 0.03])  # a plausible softmax row
    norm = normalize_attention(raw)

    assert norm.max() == pytest.approx(1.0)
    assert (norm >= ATTENTION_THRESHOLD).sum() == 1  # only the peak clears 0.6
    assert norm[2] == pytest.approx(1.0)


def test_normalize_attention_survives_all_zeros():
    assert np.all(normalize_attention(np.zeros(4)) == 0)


def test_relative_to_uniform_is_human_readable():
    """`w * T` -> "3.2x more attention than average". 1.0 means average."""
    uniform = np.full(16, 1 / 16)
    assert attention_relative_to_uniform(uniform) == pytest.approx(np.ones(16))

    peaked = np.array([0.2] + [0.8 / 15] * 15)
    assert attention_relative_to_uniform(peaked)[0] == pytest.approx(3.2)


# --------------------------------------------------------------------------- #
# T50: timeline
# --------------------------------------------------------------------------- #
def test_timeline_has_real_timestamps():
    """Needs T40's fps + source_indices; without them the spec's "0-2s / 2-5s"
    timeline cannot exist."""
    frames = build_timeline(
        p_fake=np.array([0.1, 0.9]),
        source_indices=[0, 300],
        fps=30.0,
    )
    assert frames[0].t_seconds == pytest.approx(0.0)
    assert frames[1].t_seconds == pytest.approx(10.0)


def test_timeline_reports_none_when_fps_is_unavailable():
    """A faked timestamp is worse than a missing one."""
    frames = build_timeline(p_fake=np.array([0.1, 0.9]), fps=0.0)
    assert all(f.t_seconds is None for f in frames)


def test_a_frame_needs_both_high_score_and_high_attention():
    """A high-p_fake frame the temporal branch ignored is not what the timeline
    claims to show, and vice versa."""
    p_fake = np.array([0.9, 0.9, 0.2])
    attention = np.array([0.10, 0.01, 0.10])  # frame 1 is barely attended

    frames = build_timeline(p_fake, attention=attention)
    assert frames[0].suspicious          # high score AND attended
    assert not frames[1].suspicious      # high score, ignored
    assert not frames[2].suspicious      # attended, low score


def test_interpolated_frames_are_never_suspicious():
    """A copied face is not evidence.

    _interpolate_missing fills a faceless frame from a neighbour. Flagging that
    as a detection is asserting something that was never measured.
    """
    frames = build_timeline(
        p_fake=np.array([0.99, 0.99]),
        attention=np.array([0.5, 0.5]),
        interpolated=[False, True],
    )
    assert frames[0].suspicious
    assert not frames[1].suspicious, "an interpolated frame was flagged as evidence"
    assert frames[1].interpolated


def test_spans_need_consecutive_frames():
    """One isolated sample is not a region.

    16 samples across a whole video can be ~19s apart, so a lone point says
    nothing about a span of time.
    """
    frames = build_timeline(
        p_fake=np.array([0.9, 0.1, 0.9, 0.9]),
        attention=np.array([0.25] * 4),
        source_indices=[0, 30, 60, 90],
        fps=30.0,
    )
    spans = merge_spans(frames)
    assert len(spans) == 1, "the isolated frame 0 should not become a span"
    assert spans[0].n_frames == 2
    assert spans[0].start_s == pytest.approx(2.0)


def test_no_spans_when_nothing_is_suspicious():
    frames = build_timeline(p_fake=np.array([0.1] * 4), attention=np.array([0.25] * 4))
    assert merge_spans(frames) == []


def test_timeline_description_always_states_the_sampling_caveat():
    """"2-5s suspicious" implies continuous analysis. It was 16 samples."""
    frames = build_timeline(p_fake=np.array([0.9, 0.9]), attention=np.array([0.5, 0.5]),
                            source_indices=[0, 30], fps=30.0)
    lines = describe_timeline(frames, merge_spans(frames))
    assert any("sampled across the video" in line for line in lines)
    assert any("not a" in line and "continuous" in line for line in lines)


def test_description_flags_interpolated_frames():
    frames = build_timeline(p_fake=np.array([0.9, 0.9]), interpolated=[False, True])
    lines = describe_timeline(frames, [])
    assert any("not independent evidence" in line for line in lines)


def test_per_frame_scores_need_a_video_model():
    from ml.models.classifier import ImageClassifier

    model = ImageClassifier(pretrained=False).eval()
    with pytest.raises(ValueError, match=r"image model has no timeline"):
        per_frame_scores(model, {"spatial": None, "frequency": None})


@pytest.mark.slow
def test_per_frame_scores_from_a_real_clip():
    """The whole point: 16 per-frame scores for ~16 tiny MLP passes."""
    from ml.models.classifier import VideoClassifier

    torch.manual_seed(0)
    model = VideoClassifier(pretrained=False)
    model.train()  # eval() would zero the untrained spatial branch (Milestone 2)
    with torch.no_grad():
        _logits, aux = model.forward_explain(torch.randn(1, 16, 3, 224, 224))

    scores = per_frame_scores(model, aux)
    assert scores.shape == (16,)
    assert ((scores >= 0) & (scores <= 1)).all()


# --------------------------------------------------------------------------- #
# T49: frequency
# --------------------------------------------------------------------------- #
def test_high_frequency_ratio_is_a_real_ratio():
    """Log magnitudes are frequently negative; a "ratio" of signed quantities can
    exceed 1 or flip sign. Shifting to non-negative first is not cosmetic."""
    rng = np.random.default_rng(0)
    spectrum = rng.normal(-2, 3, (64, 64))  # deliberately negative-heavy
    ratio = high_frequency_ratio(spectrum)
    assert 0.0 <= ratio <= 1.0


def test_high_frequency_ratio_rises_with_high_frequency_energy():
    """The number has to actually track the thing it claims to measure."""
    size = 64
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((y - 32) ** 2 + (x - 32) ** 2)

    low = np.where(r < 16, 5.0, 0.0)    # energy concentrated at low frequency
    high = np.where(r >= 16, 5.0, 0.0)  # ...and at high

    assert high_frequency_ratio(high) > high_frequency_ratio(low)


def test_radial_profile_shape_and_finiteness():
    rng = np.random.default_rng(0)
    profile = radial_profile(rng.normal(0, 1, (64, 64)), n_bins=32)
    assert profile.shape == (32,)
    assert np.isfinite(profile).all()


def test_log_spectrum_uses_the_models_own_transform():
    """An explanation computed with a different transform than the branch uses
    would describe a spectrum the model never saw."""
    image = torch.randn(3, 224, 224)
    spectrum = log_spectrum(image)
    assert spectrum.shape == (224, 224)
    assert np.isfinite(spectrum).all()


def test_frequency_evidence_is_serialisable():
    evidence, spectrum = frequency_evidence(torch.randn(3, 224, 224))
    payload = evidence.to_dict()

    import json

    json.dumps(payload)  # raises on numpy scalars
    assert 0.0 <= payload["hf_energy_ratio"] <= 1.0
    assert len(payload["radial_profile"]) > 0
    assert spectrum.shape == (224, 224)


def test_frequency_description_hedges_without_a_reference():
    """An absolute HF ratio means nothing without knowing what normal looks like.
    Quoting one as if it did is false precision."""
    evidence, _ = frequency_evidence(torch.randn(3, 224, 224))
    lines = describe_frequency(evidence)
    assert any("No reference profile" in line for line in lines)


def test_frequency_description_compares_when_a_reference_exists():
    evidence, _ = frequency_evidence(
        torch.randn(3, 224, 224),
        references={"real": np.linspace(5, 1, 64).tolist()},
    )
    lines = describe_frequency(evidence)
    assert not any("No reference profile" in line for line in lines)
    assert any("typical real face" in line for line in lines)
