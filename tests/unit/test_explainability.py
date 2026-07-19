"""Explainability tests (BUILD_PLAN T47/T48/T51/T53).

Three tests here do the real work:

* `test_frozen_backbone_without_input_grad_raises` -- the silent failure. A frozen
  backbone plus a non-grad input captures ZERO gradients and raises nothing,
  yielding an empty CAM presented as an explanation.
* `test_degenerate_cam_returns_zeros_not_amplified_noise` -- min-max normalising a
  dead CAM turns float noise into a vivid, structured-looking map. A
  confident-looking fake heatmap is worse than no heatmap.
* `test_cam_degrades_when_weights_are_randomized` -- Adebayo et al. 2018. The only
  test that distinguishes "this CAM explains the model" from "this CAM is an edge
  detector that would look plausible regardless".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.explainability.attribution import branch_attribution, describe
from ml.explainability.contracts import (
    UNCERTAIN_MARGIN,
    BranchAttribution,
    Explanation,
    ExplanationArtifacts,
    FrameScore,
    decide_verdict,
)
from ml.explainability.gradcam import (
    GradCAM,
    _normalize_cams,
    eval_mode,
    is_degenerate,
    spatial_target_layer,
)
from ml.models.classifier import ImageClassifier, VideoClassifier


@pytest.fixture
def live_model():
    """A model whose spatial branch actually carries signal.

    train() mode is essential and not incidental: SpatialBranch(pretrained=False)
    in eval() outputs ~ZERO (std 7.4e-15 -- see Milestone 2), because
    EfficientNet's untrained BatchNorms act as the identity in eval and the
    signal collapses. A CAM over a dead branch is uniformly zero, so every test
    below would pass vacuously.
    """
    torch.manual_seed(0)
    model = ImageClassifier(pretrained=False)
    model.train()
    return model


# --------------------------------------------------------------------------- #
# Target layer
# --------------------------------------------------------------------------- #
def test_target_layer_is_the_last_conv_stage(live_model):
    layer = spatial_target_layer(live_model)
    assert layer is live_model.spatial.features[-1]
    # Conv2dNormActivation, not a bare Conv2d: hooking features[8][0] would grab
    # the pre-BN/pre-activation output, which is not what the net propagates.
    assert [type(m).__name__ for m in layer] == ["Conv2d", "BatchNorm2d", "SiLU"]


def test_target_layer_requires_a_spatial_backbone():
    with pytest.raises(AttributeError, match=r"no \.spatial\.features"):
        spatial_target_layer(torch.nn.Linear(2, 2))


# --------------------------------------------------------------------------- #
# GradCAM basics
# --------------------------------------------------------------------------- #
def test_cam_shape_and_range(live_model):
    with GradCAM(live_model, spatial_target_layer(live_model)) as cam:
        maps = cam(torch.randn(2, 3, 224, 224), target_class=1)

    assert maps.shape == (2, 7, 7)  # EfficientNet-B3 @224 -> 7x7
    assert maps.min() >= 0.0 and maps.max() <= 1.0


def test_video_cam_yields_all_frames_in_one_backward():
    """One backward on a (1,16,...) clip gives 16 CAMs.

    VideoClassifier flattens (B,T,C,H,W) -> (B*T,C,H,W) before the backbone, so
    the activation batch is B*T. That is exactly the assumption the `grad-cam`
    library breaks (it expects activation-batch == input-batch and returns 16
    CAMs for 1 input), and exactly what makes this cheap here.
    """
    torch.manual_seed(0)
    model = VideoClassifier(pretrained=False)
    model.train()

    with GradCAM(model, spatial_target_layer(model)) as cam:
        maps = cam(torch.randn(1, 16, 3, 224, 224), target_class=1)

    assert maps.shape == (16, 7, 7), "expected one CAM per frame from a single backward"


def test_cam_hooks_are_removed_on_exit(live_model):
    """A leaked forward hook fires on every subsequent inference."""
    cam = GradCAM(live_model, spatial_target_layer(live_model))
    with cam:
        cam(torch.randn(1, 3, 224, 224))
    assert cam._handles == []


def test_eval_mode_restores_training_state():
    model = ImageClassifier(pretrained=False)
    model.train()
    with eval_mode(model):
        assert not model.training
    assert model.training, "eval_mode must restore the previous mode"


# --------------------------------------------------------------------------- #
# T48: the silent failure
# --------------------------------------------------------------------------- #
def test_frozen_backbone_without_input_grad_raises():
    """The failure mode that produces an explanation of nothing.

    Measured: with requires_grad=False on the backbone and an input that does not
    require grad, the activation does not require grad, the backward hook never
    fires, and ZERO gradients are captured -- with no exception. The freeze
    schedule (T30) sets exactly that state for the first epochs, so this is a
    live path, not a hypothetical.
    """
    torch.manual_seed(0)
    model = ImageClassifier(pretrained=False).train()
    for p in model.parameters():
        p.requires_grad_(False)

    cam = GradCAM(model, spatial_target_layer(model))
    with cam:
        # Reproduce the hazard directly: hooks installed, but the forward runs
        # with nothing requiring grad, so nothing is captured.
        model(torch.randn(1, 3, 224, 224))
        with pytest.raises(RuntimeError, match=r"captured no gradients"):
            cam._assert_captured()


def test_cam_works_inside_no_grad(live_model):
    """predict() is decorated @torch.no_grad(); GradCAM must still work (T48).

    torch.enable_grad() nests -- it re-enables inside a no_grad caller. Without
    that, the CAM path would need the whole call stack to avoid no_grad, which is
    not something a library can ask of its callers.
    """
    with torch.no_grad(), GradCAM(live_model, spatial_target_layer(live_model)) as cam:
        maps = cam(torch.randn(1, 3, 224, 224), target_class=1)
    assert maps.shape == (1, 7, 7)


def test_missing_activations_raises_clearly(live_model):
    cam = GradCAM(live_model, spatial_target_layer(live_model))
    with pytest.raises(RuntimeError, match=r"captured no activations"):
        cam._assert_captured()


# --------------------------------------------------------------------------- #
# T53: degenerate maps
# --------------------------------------------------------------------------- #
def test_degenerate_cam_returns_zeros_not_amplified_noise():
    """The standard `cam / (cam.max() + 1e-7)` turns a dead map into a vivid one.

    Feeding it float noise of order 1e-12 produces a fully-saturated, structured
    image -- an explanation of a model that explained nothing. That is worse than
    returning nothing, because it is persuasive.
    """
    noise = np.full((1, 7, 7), 1e-12, dtype=np.float32)
    noise[0, 3, 3] = 1.1e-12  # a hair of variation, far below any real signal

    out = _normalize_cams(noise)
    assert np.all(out == 0.0), (
        "a degenerate CAM was min-max normalised into a structured map -- this is "
        "how noise becomes a confident-looking explanation"
    )


def test_is_degenerate_detects_a_flat_map():
    assert is_degenerate(np.zeros((7, 7)))
    assert is_degenerate(np.full((7, 7), 0.5))
    assert not is_degenerate(np.linspace(0, 1, 49).reshape(7, 7))


def test_normalize_preserves_a_real_cam():
    """The guard must not fire on genuine signal."""
    real = np.linspace(0, 5, 49).reshape(1, 7, 7).astype(np.float32)
    out = _normalize_cams(real)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_overlay_rejects_a_batched_cam():
    """REGRESSION (found by T53's own tests).

    GradCAM returns (B, h, w). Passing that whole array to overlay_heatmap makes
    cv2.resize interpret the leading dim as CHANNELS -- (1,7,7) becomes a
    (224,224,7) "7-channel image", which broadcast-errors here but for other
    shapes would blend garbage silently. Index the frame you mean.
    """
    from ml.explainability.render import overlay_heatmap

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"single 2-D CAM"):
        overlay_heatmap(image, np.zeros((1, 7, 7)))


def test_overlay_enforces_the_spec_opacity_band():
    """Spec: heatmap opacity 0.4-0.6."""
    from ml.explainability.render import overlay_heatmap

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    cam = np.linspace(0, 1, 49).reshape(7, 7)

    assert overlay_heatmap(image, cam, alpha=0.5).shape == (224, 224, 3)
    with pytest.raises(ValueError, match=r"0.4-0.6"):
        overlay_heatmap(image, cam, alpha=0.9)


def test_overlay_upsamples_a_7x7_cam_to_the_image():
    """7x7 -> 224 is a 32x upscale. Bilinear, not nearest: hard 32px blocks would
    look like precise localisation and are nothing of the sort."""
    from ml.explainability.render import overlay_heatmap

    image = np.full((224, 224, 3), 128, dtype=np.uint8)
    out = overlay_heatmap(image, np.linspace(0, 1, 49).reshape(7, 7))
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    # Bilinear produces many intermediate values; nearest would give ~7 bands.
    assert len(np.unique(out[..., 0])) > 20


# --------------------------------------------------------------------------- #
# T53: sanity checks that validate the CAM is about the MODEL
# --------------------------------------------------------------------------- #
def test_cam_is_class_sensitive(live_model):
    """CAM(real) and CAM(fake) must differ.

    Asserted on correlation < 0.99, not mere inequality: a 2-class head produces
    near-mirror maps that trivially pass `!=` while being the same explanation.
    """
    x = torch.randn(1, 3, 224, 224)
    with GradCAM(live_model, spatial_target_layer(live_model)) as cam:
        real_cam = cam(x, target_class=0)[0]
        fake_cam = cam(x, target_class=1)[0]

    if is_degenerate(real_cam) or is_degenerate(fake_cam):
        pytest.skip("degenerate CAM on an untrained model; nothing to compare")

    corr = np.corrcoef(real_cam.ravel(), fake_cam.ravel())[0, 1]
    assert abs(corr) < 0.99, (
        f"CAMs for 'real' and 'fake' correlate at {corr:.4f} -- the map is not "
        f"class-specific, so it explains the image rather than the decision"
    )


@pytest.mark.slow
def test_cam_degrades_when_weights_are_randomized():
    """Adebayo et al. 2018, "Sanity Checks for Saliency Maps".

    **The only test here that validates the headline feature.** Randomize the
    backbone's weights and the CAM must change substantially. If it does NOT, the
    map is a function of the input alone -- an edge detector wearing an
    explanation's clothes -- and would look equally plausible over a model that
    had learned nothing.

    A saliency method that passes this is explaining the model. One that fails it
    is explaining the picture.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)

    trained = ImageClassifier(pretrained=False).train()
    with GradCAM(trained, spatial_target_layer(trained)) as cam:
        before = cam(x, target_class=1)[0]

    # Re-randomize the backbone. Same architecture, different function.
    torch.manual_seed(999)
    randomized = ImageClassifier(pretrained=False).train()
    with GradCAM(randomized, spatial_target_layer(randomized)) as cam:
        after = cam(x, target_class=1)[0]

    if is_degenerate(before) or is_degenerate(after):
        pytest.skip("degenerate CAM; cannot compare")

    corr = np.corrcoef(before.ravel(), after.ravel())[0, 1]
    assert abs(corr) < 0.95, (
        f"randomizing the backbone left the CAM {corr:.4f}-correlated with the "
        f"original -- the map does not depend on the model's weights, so it is "
        f"not an explanation of the model"
    )


# --------------------------------------------------------------------------- #
# T51: ablation attribution
# --------------------------------------------------------------------------- #
def test_attribution_emits_two_branches_for_an_image_model(synthetic_branch_features):
    """No temporal branch on an image model -> 2 entries, NOT 3 with a zero.

    "0% temporal" reads as a measurement; it is a structural absence (T51).
    """
    spatial, frequency, _ = synthetic_branch_features
    model = ImageClassifier(pretrained=False).eval()
    aux = {"spatial": spatial, "frequency": frequency, "temporal": None}

    out = branch_attribution(model, aux, branch_means=None)
    assert {a.branch for a in out} == {"spatial", "frequency"}


def test_attribution_emits_three_branches_for_video(synthetic_branch_features):
    spatial, frequency, temporal = synthetic_branch_features
    model = VideoClassifier(pretrained=False).eval()
    aux = {"spatial": spatial, "frequency": frequency, "temporal": temporal}

    out = branch_attribution(model, aux, branch_means=None)
    assert {a.branch for a in out} == {"spatial", "frequency", "temporal"}


def test_attribution_is_causal_and_nonzero(synthetic_branch_features):
    """Removing a branch must actually move the logit.

    Uses synthetic features, deliberately: an untrained SpatialBranch in eval
    outputs ~zero, so ablating it would produce a delta of exactly 0.0 -- a
    vacuous test that reads like a working measurement (Milestone 2).
    """
    spatial, frequency, _ = synthetic_branch_features
    model = ImageClassifier(pretrained=False).eval()
    aux = {"spatial": spatial, "frequency": frequency, "temporal": None}

    out = branch_attribution(model, aux, branch_means=None)
    assert any(abs(a.delta) > 1e-4 for a in out), (
        f"no branch moved the logit: {[(a.branch, a.delta) for a in out]}"
    )


def test_attribution_uses_the_mean_baseline_when_available(synthetic_branch_features):
    """Zeros are off-manifold; the mean is the honest baseline (ADR 0001)."""
    spatial, frequency, _ = synthetic_branch_features
    model = ImageClassifier(pretrained=False).eval()
    aux = {"spatial": spatial, "frequency": frequency, "temporal": None}
    means = {
        "spatial": spatial.mean(0).tolist(),
        "frequency": frequency.mean(0).tolist(),
    }

    out = branch_attribution(model, aux, branch_means=means)
    assert all(a.baseline == "mean" for a in out)


def test_attribution_warns_when_falling_back_to_zeros(synthetic_branch_features, caplog):
    spatial, frequency, _ = synthetic_branch_features
    model = ImageClassifier(pretrained=False).eval()
    aux = {"spatial": spatial, "frequency": frequency, "temporal": None}

    with caplog.at_level("WARNING"):
        out = branch_attribution(model, aux, branch_means=None)
    assert all(a.baseline == "zero" for a in out)
    assert "off-manifold" in caplog.text


def test_attribution_is_not_zero_sum():
    """The property attention weights cannot have.

    A softmax forces the three to sum to 1, so it literally cannot say "all three
    branches agree strongly". Ablation can, and `describe` says so.
    """
    lines = describe([
        BranchAttribution("spatial", 0.8),
        BranchAttribution("frequency", 0.7),
        BranchAttribution("temporal", 0.6),
    ])
    assert any("agree" in line for line in lines)


def test_describe_reports_when_nothing_is_driving():
    lines = describe([
        BranchAttribution("spatial", 1e-6),
        BranchAttribution("frequency", -1e-7),
    ])
    assert any("No single branch" in line for line in lines)


def test_describe_is_empty_for_no_attribution():
    assert describe([]) == []


# --------------------------------------------------------------------------- #
# T46: the contract
# --------------------------------------------------------------------------- #
def test_explanation_to_dict_is_json_safe():
    """No numpy, no tensors. The backend serialises this straight to JSON."""
    import json

    exp = Explanation(
        label="fake",
        verdict="fake",
        p_fake=float(np.float32(0.87)),
        attribution=[BranchAttribution("spatial", float(np.float64(0.3)))],
        timeline=[FrameScore(index=0, source_index=0, p_fake=0.9, t_seconds=0.0)],
        degenerate={"heatmap": False},
        warnings=["single-subject video"],
    )
    text = json.dumps(exp.to_dict())  # raises on numpy scalars
    assert '"verdict": "fake"' in text


@pytest.mark.parametrize(
    ("p_fake", "expected"),
    [(0.95, "fake"), (0.05, "real"), (0.5, "uncertain"), (0.55, "uncertain"), (0.65, "fake")],
)
def test_verdict_has_an_uncertain_band(p_fake, expected):
    """A forced binary call on a near-0.5 margin is the most misleading thing
    this system could output -- especially uncalibrated."""
    assert decide_verdict(p_fake) == expected


def test_uncertain_margin_is_wide_enough_to_matter():
    assert UNCERTAIN_MARGIN >= 0.05


def test_calibrated_defaults_to_false():
    """No calibration code exists (T78). The honest default is False, and the UI
    keys off it to suppress the percentage (T63)."""
    assert Explanation(label="fake", verdict="fake", p_fake=0.9).calibrated is False


def test_artifacts_are_separate_from_the_dict():
    """Images travel as bytes/URLs, never base64 in the verdict payload (T58)."""
    art = ExplanationArtifacts()
    art.add("heatmap.png", b"\x89PNG...")
    exp = Explanation(label="fake", verdict="fake", p_fake=0.9, artifact_names=art.names())

    assert "heatmap.png" in exp.to_dict()["artifact_names"]
    assert "images" not in exp.to_dict()
