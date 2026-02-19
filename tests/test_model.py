"""Tests for model components."""

import pytest
import torch

from adaptive_noise_schedule_diffusion_with_clip_guidance.models.model import (
    NoiseSchedulePredictor,
    AdaptiveNoiseDiffusionModel,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.models.components import (
    AdaptiveScheduleLoss,
    CLIPGuidedLoss,
    PreferenceRewardLoss,
    GatedFusion,
    TimeEmbedding,
)


class TestComponents:
    """Test custom components."""

    def test_time_embedding(self):
        """Test time embedding layer."""
        time_embed = TimeEmbedding(dim=128)

        timesteps = torch.tensor([0, 100, 500, 999])
        embeddings = time_embed(timesteps)

        assert embeddings.shape == (4, 128)

    def test_gated_fusion(self):
        """Test gated fusion mechanism."""
        fusion = GatedFusion(dim=256)

        feature_a = torch.randn(4, 256)
        feature_b = torch.randn(4, 256)

        fused = fusion(feature_a, feature_b)

        assert fused.shape == (4, 256)

    def test_adaptive_schedule_loss(self):
        """Test adaptive schedule loss."""
        loss_fn = AdaptiveScheduleLoss()

        predicted = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        clip_score = torch.tensor([0.25, 0.30])
        num_steps = torch.tensor([30, 35])

        loss, loss_dict = loss_fn(predicted, target, clip_score, num_steps)

        assert loss.item() > 0
        assert "loss" in loss_dict
        assert "recon_loss" in loss_dict
        assert "semantic_loss" in loss_dict
        assert "efficiency_loss" in loss_dict

    def test_clip_guided_loss(self):
        """Test CLIP guided loss."""
        loss_fn = CLIPGuidedLoss()

        clip_score = torch.tensor([0.25, 0.30])

        loss = loss_fn(clip_score, use_guidance=True)

        assert isinstance(loss.item(), float)

    def test_preference_reward_loss(self):
        """Test preference reward loss."""
        loss_fn = PreferenceRewardLoss()

        preferred = torch.tensor([0.8, 0.7])
        rejected = torch.tensor([0.5, 0.4])

        loss, loss_dict = loss_fn(preferred, rejected)

        assert loss.item() >= 0
        assert "preference_loss" in loss_dict
        assert "ranking_loss" in loss_dict


class TestNoiseSchedulePredictor:
    """Test noise schedule predictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = NoiseSchedulePredictor(
            clip_dim=512,
            hidden_dim=256,
            num_layers=3,
        )

        assert predictor.clip_dim == 512
        assert predictor.hidden_dim == 256

    def test_forward_pass(self, device):
        """Test forward pass."""
        predictor = NoiseSchedulePredictor(
            clip_dim=512,
            hidden_dim=128,
            num_layers=2,
        ).to(device)

        batch_size = 4
        clip_features = torch.randn(batch_size, 512).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

        outputs = predictor(clip_features, timesteps)

        assert "skip_prob" in outputs
        assert "noise_scale" in outputs
        assert outputs["skip_prob"].shape == (batch_size,)
        assert outputs["noise_scale"].shape == (batch_size,)

        # Check output ranges
        assert (outputs["skip_prob"] >= 0).all()
        assert (outputs["skip_prob"] <= 1).all()
        assert (outputs["noise_scale"] >= 0).all()
        assert (outputs["noise_scale"] <= 1).all()


class TestAdaptiveNoiseDiffusionModel:
    """Test adaptive noise diffusion model."""

    @pytest.mark.slow
    def test_model_initialization(self, device):
        """Test model initialization."""
        model = AdaptiveNoiseDiffusionModel(
            diffusion_model_id="runwayml/stable-diffusion-v1-5",
            clip_model_id="openai/clip-vit-base-patch32",
            predictor_hidden_dim=128,
            predictor_num_layers=2,
            use_adaptive_schedule=True,
            device=str(device),
        )

        assert model.use_adaptive_schedule
        assert model.schedule_predictor is not None

    @pytest.mark.slow
    def test_encode_prompt(self, device):
        """Test prompt encoding."""
        model = AdaptiveNoiseDiffusionModel(
            predictor_hidden_dim=128,
            predictor_num_layers=2,
            device=str(device),
        )

        prompts = ["a cat", "a dog"]
        embeddings = model.encode_prompt(prompts)

        assert embeddings.shape[0] == 2
        assert embeddings.ndim == 3

    @pytest.mark.slow
    def test_clip_score(self, device, sample_images, sample_captions):
        """Test CLIP score computation."""
        model = AdaptiveNoiseDiffusionModel(
            predictor_hidden_dim=128,
            predictor_num_layers=2,
            device=str(device),
        )

        images = sample_images.to(device)
        scores = model.get_clip_score(images, sample_captions)

        assert scores.shape == (2,)
        assert (scores >= 0).all()
        assert (scores <= 1).all()
