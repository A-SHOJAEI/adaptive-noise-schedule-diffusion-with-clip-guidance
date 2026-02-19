"""Tests for training components."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from adaptive_noise_schedule_diffusion_with_clip_guidance.models.model import (
    AdaptiveNoiseDiffusionModel,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.training.trainer import (
    DiffusionTrainer,
)


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, num_samples=10, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        image = (image - image.min()) / (image.max() - image.min())
        image = (image - 0.5) * 2

        return {
            "image": image,
            "caption": f"sample caption {idx}",
        }


@pytest.fixture
def mock_dataloaders():
    """Create mock dataloaders for testing."""
    train_dataset = MockDataset(num_samples=10, image_size=64)
    val_dataset = MockDataset(num_samples=5, image_size=64)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    return train_loader, val_loader


class TestTrainer:
    """Test trainer functionality."""

    @pytest.mark.slow
    def test_trainer_initialization(self, device, sample_config, mock_dataloaders):
        """Test trainer initialization."""
        train_loader, val_loader = mock_dataloaders

        model = AdaptiveNoiseDiffusionModel(
            predictor_hidden_dim=128,
            predictor_num_layers=2,
            use_adaptive_schedule=True,
            device=str(device),
        )

        trainer = DiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None

    @pytest.mark.slow
    def test_train_epoch(self, device, sample_config, mock_dataloaders):
        """Test single training epoch."""
        train_loader, val_loader = mock_dataloaders

        # Modify config for fast testing
        sample_config["training"]["num_epochs"] = 1
        sample_config["training"]["mixed_precision"] = False

        model = AdaptiveNoiseDiffusionModel(
            predictor_hidden_dim=128,
            predictor_num_layers=2,
            use_adaptive_schedule=True,
            device=str(device),
        )

        trainer = DiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
        )

        metrics = trainer.train_epoch(epoch=1)

        assert "train_loss" in metrics
        assert "learning_rate" in metrics
        assert isinstance(metrics["train_loss"], float)

    @pytest.mark.slow
    def test_validation(self, device, sample_config, mock_dataloaders):
        """Test validation."""
        train_loader, val_loader = mock_dataloaders

        model = AdaptiveNoiseDiffusionModel(
            predictor_hidden_dim=128,
            predictor_num_layers=2,
            use_adaptive_schedule=True,
            device=str(device),
        )

        trainer = DiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
        )

        metrics = trainer.validate(epoch=1)

        assert "val_loss" in metrics
        assert "val_clip_score" in metrics
        assert isinstance(metrics["val_loss"], float)

    @pytest.mark.slow
    def test_checkpoint_saving(self, device, sample_config, mock_dataloaders, temp_output_dir):
        """Test checkpoint saving."""
        train_loader, val_loader = mock_dataloaders

        # Update checkpoint dir
        sample_config["checkpoint"]["save_dir"] = str(temp_output_dir / "checkpoints")

        model = AdaptiveNoiseDiffusionModel(
            predictor_hidden_dim=128,
            predictor_num_layers=2,
            use_adaptive_schedule=True,
            device=str(device),
        )

        trainer = DiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sample_config,
            device=device,
        )

        # Save checkpoint
        metrics = {"train_loss": 1.0, "val_loss": 1.0}
        trainer.save_checkpoint(epoch=1, metrics=metrics, is_best=True)

        # Check files exist
        checkpoint_dir = temp_output_dir / "checkpoints"
        assert (checkpoint_dir / "best_model.pt").exists()
