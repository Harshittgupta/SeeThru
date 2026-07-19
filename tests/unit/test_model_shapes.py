"""Model shape/contract tests (BUILD_PLAN T14).

Lifted from the `if __name__ == "__main__":` demo blocks that previously lived in
classifier.py, fusion.py, spatial_branch.py, frequency_branch.py and
temporal_branch.py. Those asserts were real checks that nobody ever ran -- they
only fired if a human happened to execute the module directly. Now CI runs them.

`pretrained=False` throughout: that is what keeps the suite offline (no
torchvision weight download) and CPU-only. Random weights exercise shapes and
gradient flow perfectly well.
"""

from __future__ import annotations

import pytest
import torch

from ml.models.branches import FrequencyBranch, SpatialBranch, TemporalBranch
from ml.models.classifier import ImageClassifier, VideoClassifier
from ml.models.fusion import (
    FREQUENCY_DIM,
    FUSED_OUTPUT_DIM,
    SPATIAL_DIM,
    TEMPORAL_DIM,
    AttentionFusion,
    FeatureFusion,
)


# --------------------------------------------------------------------------- #
# Branches
# --------------------------------------------------------------------------- #
def test_spatial_branch_shape():
    model = SpatialBranch(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, SPATIAL_DIM)
    assert model.out_features == SPATIAL_DIM


def test_frequency_branch_shape():
    model = FrequencyBranch().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, FREQUENCY_DIM)


def test_frequency_branch_gradient_flows_through_fft():
    """The FFT must stay differentiable, and finite.

    `log(|fft| + 1e-8)` is an exponent-range hazard: a constant image makes every
    non-DC bin exactly zero, so a naive implementation NaNs here.
    """
    model = FrequencyBranch()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "non-finite gradient through the FFT"


def test_frequency_branch_survives_a_constant_image():
    """The degenerate case for log-magnitude: all non-DC bins are exactly 0."""
    model = FrequencyBranch()
    x = torch.full((1, 3, 224, 224), 0.5, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()


def test_temporal_branch_shape_and_attention():
    model = TemporalBranch(input_size=SPATIAL_DIM).eval()
    with torch.no_grad():
        pooled, weights = model(torch.randn(4, 16, SPATIAL_DIM), return_attention=True)
    assert pooled.shape == (4, TEMPORAL_DIM)
    assert weights.shape == (4, 16)
    # Softmax over time: rows sum to 1. This is exactly why the spec's raw 0.6
    # attention threshold can never fire at T=16 (uniform = 0.0625) -- see T50.
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fusion_cls", [FeatureFusion, AttentionFusion])
def test_fusion_shape_with_and_without_temporal(fusion_cls):
    model = fusion_cls()
    model.train()  # exercise the norm layers in training mode
    spatial = torch.randn(4, SPATIAL_DIM)
    frequency = torch.randn(4, FREQUENCY_DIM)
    temporal = torch.randn(4, TEMPORAL_DIM)

    assert model(spatial, frequency, temporal).shape == (4, FUSED_OUTPUT_DIM)
    assert model(spatial, frequency).shape == (4, FUSED_OUTPUT_DIM)


@pytest.mark.parametrize("fusion_cls", [FeatureFusion, AttentionFusion])
def test_fusion_survives_batch_of_one_in_train_mode(fusion_cls):
    """A trailing batch of 1 must not crash the fusion MLP. REGRESSION TEST (T21).

    DataLoader defaults to drop_last=False, so a batch of 1 is inevitable at some
    random epoch boundary -- meaning this killed a training run at a random point,
    hours in. Reproduced live 2026-07-15, before the fix:

        ValueError: Expected more than 1 value per channel when training,
        got input size torch.Size([1, 512])

    Fixed by T21 (BatchNorm1d -> LayerNorm). Note that drop_last=True would have
    hidden this without fixing the real problem: at the spec's batch size of 8-16,
    BatchNorm's statistics are estimated from 8-16 samples, which is noise. And
    gradient accumulation does not help -- BN normalizes per *micro*-batch, so
    accumulating 2x8 still estimates from 8.
    """
    model = fusion_cls()
    model.train()
    out = model(torch.randn(1, SPATIAL_DIM), torch.randn(1, FREQUENCY_DIM))
    assert out.shape == (1, FUSED_OUTPUT_DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("fusion_cls", [FeatureFusion, AttentionFusion])
def test_fusion_has_no_batch_dependent_norm(fusion_cls):
    """No BatchNorm anywhere in fusion, and no running statistics (T21).

    Guards the *reason*, not just the symptom. Someone could "fix" the batch-of-1
    crash with drop_last=True and reintroduce BatchNorm here, and every other test
    would still pass while training silently normalized over 8-sample noise.
    """
    offenders = [
        (name, type(m).__name__)
        for name, m in fusion_cls().named_modules()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
    ]
    assert not offenders, (
        f"batch-dependent norm found in fusion: {offenders}. "
        f"See docs/BUILD_PLAN.md T21 -- batch size here is 8-16, where BN "
        f"statistics are noise, and gradient accumulation does not fix that."
    )


@pytest.mark.parametrize("fusion_cls", [FeatureFusion, AttentionFusion])
def test_fusion_output_is_batch_size_independent(fusion_cls):
    """The same sample must fuse identically alone or inside a batch (T21).

    This is the property LayerNorm buys and BatchNorm cannot: under BN in train
    mode, a sample's output depended on whichever samples happened to share its
    batch -- so the same image scored differently depending on batch composition.

    dropout=0.0 is essential here, not incidental: dropout is also active in
    train() mode and is random per call, so with it enabled this test fails even
    under LayerNorm and tells you nothing about the norm. (Learned the hard way.)
    We need train() specifically because eval() is exactly where BatchNorm
    switches to running statistics and *would* look batch-independent -- hiding
    the bug we are testing for.
    """
    model = fusion_cls(dropout=0.0)
    model.train()  # the mode where BN would differ; dropout disabled above
    torch.manual_seed(0)
    s = torch.randn(4, SPATIAL_DIM)
    f = torch.randn(4, FREQUENCY_DIM)

    with torch.no_grad():
        batched = model(s, f)[0]
        alone = model(s[:1], f[:1])[0]
    assert torch.allclose(batched, alone, atol=1e-5), (
        "a sample's fused output depends on its batch-mates -- "
        "is a batch-dependent norm back?"
    )


@pytest.mark.parametrize("fusion_cls", [FeatureFusion, AttentionFusion])
def test_dropout_reaches_the_fusion_mlp(fusion_cls):
    """The dropout argument must actually apply (T22).

    It previously reached only the classifier's final head while the fusion MLP
    stayed hardcoded at 0.4 -- so the spec's "dropout 0.3-0.5" was unreachable
    from config, and the 1.2M-parameter fusion MLP (where most of the overfitting
    risk lives) ignored the knob entirely. A config option that silently does
    nothing is worse than no option: you tune it, see no effect, and conclude
    dropout doesn't help.
    """
    rates = {
        m.p for m in fusion_cls(dropout=0.25).modules() if isinstance(m, torch.nn.Dropout)
    }
    assert rates == {0.25}, f"fusion dropout did not take effect: found {rates}"


def test_classifier_threads_dropout_all_the_way_down():
    """End-to-end: DeepfakeClassifier(dropout=x) must reach fusion AND the head."""
    model = ImageClassifier(pretrained=False, dropout=0.33)
    rates = {m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}
    assert 0.33 in rates
    assert rates == {0.33}, (
        f"some Dropout layers ignored the constructor argument: {rates}"
    )


def test_attention_fusion_weights_sum_to_one():
    model = AttentionFusion().eval()
    _, weights = model(
        torch.randn(4, SPATIAL_DIM),
        torch.randn(4, FREQUENCY_DIM),
        torch.randn(4, TEMPORAL_DIM),
        return_weights=True,
    )
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_attention_fusion_masks_absent_temporal_to_exactly_zero():
    """Without temporal features its weight must be 0, not merely small.

    The frontend renders this: "0% temporal" would read as a measurement, when it
    is really a structural absence (T51). Also a NaN hazard -- the mask is -inf
    fed to a softmax.
    """
    model = AttentionFusion().eval()
    _, weights = model(
        torch.randn(4, SPATIAL_DIM), torch.randn(4, FREQUENCY_DIM), return_weights=True
    )
    assert torch.equal(weights[:, 2], torch.zeros(4))
    assert torch.isfinite(weights).all(), "softmax over -inf produced non-finite weights"


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
def test_image_classifier_shape(image_model):
    with torch.no_grad():
        logits = image_model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)


def test_video_classifier_shape(video_model):
    with torch.no_grad():
        logits = video_model(torch.randn(2, 16, 3, 224, 224))
    assert logits.shape == (2, 2)


def test_predict_returns_label_and_confidence(image_model):
    out = image_model.predict(torch.randn(3, 224, 224))  # unbatched
    assert out["label"] in ("real", "fake")
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["logits"].shape == (2,)


def test_video_predict_accepts_unbatched_clip(video_model):
    out = video_model.predict(torch.randn(16, 3, 224, 224))
    assert out["label"] in ("real", "fake")


def test_predict_restores_training_mode(image_model):
    """predict() flips to eval() and back. If it fails to restore, a subsequent
    training step silently runs with eval-mode norm/dropout."""
    image_model.train()
    image_model.predict(torch.randn(3, 224, 224))
    assert image_model.training, "predict() did not restore training mode"


def test_untrained_spatial_branch_is_dead_in_eval_mode():
    """PINS A TESTING HAZARD, not a bug in the model (T23/T51).

    SpatialBranch(pretrained=False).eval() emits effectively ZERO. EfficientNet's
    BatchNorms hold their initial running stats (mean=0, var=1), so in eval mode
    they are the identity, nothing rescales between layers, and the signal
    collapses on the way through. Measured:

        pretrained=False, eval   std 7.4e-15   <- dead
        pretrained=False, train  std 8.0e-02
        pretrained=True,  eval   std 8.7e-02

    Why this is worth a test: it makes any test that depends on the spatial
    branch's *values* pass vacuously. Ablating an already-zero branch yields a
    delta of exactly 0.0 -- which reads as "the test ran and the number came out"
    rather than "the test measured nothing". T51's attribution tests must use
    synthetic_branch_features or pretrained=True.

    If this test ever fails, the hazard is gone and the warnings in conftest.py
    and BUILD_PLAN T51 should be removed.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    dead = SpatialBranch(pretrained=False).eval()
    with torch.no_grad():
        assert dead(x).abs().max() < 1e-10, (
            "untrained SpatialBranch is no longer dead in eval mode -- "
            "the conftest/T51 warnings about vacuous ablation tests can go"
        )

    alive = SpatialBranch(pretrained=False).train()
    with torch.no_grad():
        assert alive(x).std() > 1e-3, "train() mode should carry signal"


def test_ablation_needs_live_features(synthetic_branch_features):
    """The T51 ablation path works when the branches actually carry signal.

    The counterpart to the test above: same machinery, real features, real delta.
    """
    from ml.models.fusion import FeatureFusion

    spatial, frequency, temporal = synthetic_branch_features
    model = ImageClassifier(pretrained=False).eval()
    assert isinstance(model.fusion, FeatureFusion)  # ADR 0001

    with torch.no_grad():
        full = model.fuse_and_classify(spatial, frequency, temporal)[:, 1]
        no_spatial = model.fuse_and_classify(
            torch.zeros_like(spatial), frequency, temporal
        )[:, 1]

    delta = (full - no_spatial).abs()
    assert (delta > 1e-4).all(), (
        f"ablating the spatial branch moved the fake logit by {delta.tolist()} -- "
        f"if this is ~0, the features are dead and the test proves nothing"
    )


def test_image_and_video_state_dicts_differ_only_by_temporal():
    """Underpins the two-stage transfer of T33.

    VideoClassifier subclasses DeepfakeClassifier, so spatial/frequency/fusion/
    classifier share module paths exactly -- which is what lets stage 2 load
    stage 1's weights with strict=False. If this drifts, transfer silently
    becomes "train from scratch" and nobody notices.
    """
    img = ImageClassifier(pretrained=False)
    vid = VideoClassifier(pretrained=False)

    missing, unexpected = vid.load_state_dict(img.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys when transferring: {unexpected}"
    assert all(k.startswith("temporal.") for k in missing), (
        f"expected only temporal.* to be missing, got: "
        f"{[k for k in missing if not k.startswith('temporal.')][:5]}"
    )
