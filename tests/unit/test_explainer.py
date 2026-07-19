"""The Explainer facade, end to end (BUILD_PLAN T46-T52).

The tests that matter are about **degrading honestly**. A failed heatmap must
become "no heatmap", never a 500 and never a map made of amplified noise -- the
verdict and the attribution are still perfectly good without it.
"""

from __future__ import annotations

import json

import pytest
import torch

from ml.explainability import Explainer
from ml.models.classifier import ImageClassifier, VideoClassifier


def _warm_batchnorm(model, shape: tuple, n: int = 12):
    """Give an untrained model sensible BatchNorm running statistics.

    Without this, every test below passes vacuously and for a subtle reason.
    `Explainer` correctly forces eval() -- dropout would otherwise randomise the
    CAM -- and eval() is exactly where an untrained EfficientNet dies: its BN
    layers still hold their INIT running stats (mean=0, var=1), so they act as
    the identity, nothing rescales between layers, and the signal collapses to
    ~zero (std 7.4e-15, Milestone 2). Every CAM comes out degenerate and is
    correctly omitted, so an assertion like "a heatmap was produced" fails, and
    an assertion like "attribution is non-zero" would pass while measuring
    nothing.

    Running a few forward passes in train() mode is precisely what training does
    to those statistics. Verified: eval CAM degenerate before = True, after =
    False. This keeps the tests fast and offline while making the model behave
    like a real one -- rather than weakening the Explainer to suit the fixture.
    """
    model.train()
    with torch.no_grad():
        for _ in range(n):
            model(torch.randn(*shape))
    return model


@pytest.fixture
def image_explainer():
    torch.manual_seed(0)
    model = _warm_batchnorm(ImageClassifier(pretrained=False), (4, 3, 224, 224))
    return Explainer(model, meta={"calibrated": False})


@pytest.fixture
def video_explainer():
    torch.manual_seed(0)
    model = _warm_batchnorm(VideoClassifier(pretrained=False), (1, 4, 3, 224, 224), n=6)
    return Explainer(model, meta={"calibrated": False})


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
def test_explain_image_produces_a_serialisable_explanation(image_explainer):
    explanation, artifacts = image_explainer.explain_image(torch.randn(1, 3, 224, 224))

    payload = explanation.to_dict()
    json.dumps(payload)  # raises on numpy scalars / tensors

    assert payload["label"] in ("real", "fake")
    assert payload["verdict"] in ("real", "fake", "uncertain")
    assert 0.0 <= payload["p_fake"] <= 1.0
    assert payload["calibrated"] is False
    assert len(payload["attribution"]) == 2  # image model: no temporal branch
    assert payload["frequency"]["hf_energy_ratio"] >= 0.0
    assert artifacts.names()


def test_image_artifacts_are_real_pngs(image_explainer):
    _explanation, artifacts = image_explainer.explain_image(torch.randn(1, 3, 224, 224))
    for name, png in artifacts.images.items():
        assert png[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not a PNG"


def test_images_are_not_in_the_json(image_explainer):
    """Artifacts travel as bytes/URLs, never base64 in the verdict payload (T58)."""
    explanation, artifacts = image_explainer.explain_image(torch.randn(1, 3, 224, 224))
    payload = json.dumps(explanation.to_dict())

    assert len(payload) < 10_000, "the explanation JSON should stay small and streamable"
    assert "heatmap.png" in explanation.to_dict()["artifact_names"]


def test_uncalibrated_warning_is_always_present(image_explainer):
    """No calibration code exists (T78), so this must fire on every run."""
    explanation, _ = image_explainer.explain_image(torch.randn(1, 3, 224, 224))
    assert any("uncalibrated" in w for w in explanation.warnings)


def test_explainer_restores_training_mode(image_explainer):
    """Explaining must not silently leave a training model in eval."""
    image_explainer.model.train()
    image_explainer.explain_image(torch.randn(1, 3, 224, 224))
    assert image_explainer.model.training


def test_explain_accepts_an_unbatched_image(image_explainer):
    explanation, _ = image_explainer.explain_image(torch.randn(3, 224, 224))
    assert explanation.verdict in ("real", "fake", "uncertain")


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_explain_clip_produces_a_timeline(video_explainer):
    explanation, artifacts = video_explainer.explain_clip(
        torch.randn(1, 8, 3, 224, 224),
        fps=30.0,
        source_indices=[0, 30, 60, 90, 120, 150, 180, 210],
    )
    payload = explanation.to_dict()
    json.dumps(payload)

    assert len(payload["timeline"]) == 8
    assert payload["timeline"][1]["t_seconds"] == pytest.approx(1.0)  # frame 30 @ 30fps
    assert len(payload["attribution"]) == 3  # video model has a temporal branch
    assert "timeline.png" in artifacts.names()


@pytest.mark.slow
def test_clip_cams_are_produced_per_frame(video_explainer):
    """One backward gives all frame CAMs -- the (B,T)->(B*T) flatten (T47)."""
    _explanation, artifacts = video_explainer.explain_clip(torch.randn(1, 4, 3, 224, 224))
    cam_names = [n for n in artifacts.names() if n.startswith("cam_f")]
    assert len(cam_names) == 4


@pytest.mark.slow
def test_interpolated_frames_reach_the_explanation(video_explainer):
    """A copied face must be marked all the way to the UI (T50)."""
    explanation, _ = video_explainer.explain_clip(
        torch.randn(1, 4, 3, 224, 224),
        fps=30.0,
        source_indices=[0, 30, 60, 90],
        interpolated=[False, False, True, False],
    )
    timeline = explanation.to_dict()["timeline"]
    assert timeline[2]["interpolated"] is True
    assert timeline[2]["suspicious"] is False, "an interpolated frame was called evidence"
    assert any("not independent evidence" in w for w in explanation.warnings)


@pytest.mark.slow
def test_clip_description_states_the_sampling_caveat(video_explainer):
    explanation, _ = video_explainer.explain_clip(torch.randn(1, 4, 3, 224, 224), fps=30.0)
    assert any("sampled across the video" in w for w in explanation.warnings)


# --------------------------------------------------------------------------- #
# Degrading honestly
# --------------------------------------------------------------------------- #
def test_a_failed_heatmap_degrades_instead_of_raising(image_explainer, monkeypatch):
    """The verdict and attribution are still good without a heatmap.

    A 500 because the *picture* failed would be a poor trade, and a heatmap made
    of noise would be worse than either.

    Breaks only the CAM path, via monkeypatch. An earlier version deleted
    `model.spatial.features`, which also breaks the forward pass -- so it was
    testing "the model is destroyed", not "the heatmap failed".
    """
    import ml.explainability.explainer as module

    def boom(_model):
        raise AttributeError("simulated: no target layer")

    monkeypatch.setattr(module, "spatial_target_layer", boom)

    explanation, artifacts = image_explainer.explain_image(torch.randn(1, 3, 224, 224))
    assert explanation.degenerate["heatmap"] is True
    assert "heatmap.png" not in artifacts.names()
    # Everything that does not depend on the CAM still works.
    assert explanation.verdict in ("real", "fake", "uncertain")
    assert len(explanation.attribution) == 2
    assert any("could not be produced" in w for w in explanation.warnings)


def test_degenerate_cam_is_reported_not_rendered():
    """The dead-branch case, which is a real state, not a contrivance.

    An untrained model in eval() has BN running stats still at init, so the
    spatial branch outputs ~zero (Milestone 2) and every CAM is flat. Min-max
    normalising a flat CAM amplifies float noise into a vivid, structured map --
    an explanation of a model that explained nothing. It must be omitted and
    reported.
    """
    torch.manual_seed(0)
    model = ImageClassifier(pretrained=False)
    model.eval()  # NOT warmed: the spatial branch outputs ~0 here
    explainer = Explainer(model, meta={"calibrated": False})

    explanation, artifacts = explainer.explain_image(torch.randn(1, 3, 224, 224))
    assert explanation.degenerate["heatmap"] is True
    assert "heatmap.png" not in artifacts.names()
    assert any("no signal" in w or "could not be produced" in w for w in explanation.warnings)
    # The verdict survives: a missing picture is not a missing prediction.
    assert explanation.verdict in ("real", "fake", "uncertain")


def test_branch_means_from_meta_are_used(synthetic_branch_features):
    """The checkpoint's branch_means make ablation on-manifold (ADR 0001)."""
    spatial, frequency, _ = synthetic_branch_features
    torch.manual_seed(0)
    model = _warm_batchnorm(ImageClassifier(pretrained=False), (4, 3, 224, 224), n=4)

    explainer = Explainer(
        model,
        meta={
            "calibrated": False,
            "branch_means": {
                "spatial": spatial.mean(0).tolist(),
                "frequency": frequency.mean(0).tolist(),
            },
        },
    )
    explanation, _ = explainer.explain_image(torch.randn(1, 3, 224, 224))
    assert all(a.baseline == "mean" for a in explanation.attribution)


def test_calibrated_meta_is_respected():
    torch.manual_seed(0)
    model = _warm_batchnorm(ImageClassifier(pretrained=False), (4, 3, 224, 224), n=4)
    explainer = Explainer(model, meta={"calibrated": True})

    explanation, _ = explainer.explain_image(torch.randn(1, 3, 224, 224))
    assert explanation.calibrated is True
    assert not any("uncalibrated" in w for w in explanation.warnings)
