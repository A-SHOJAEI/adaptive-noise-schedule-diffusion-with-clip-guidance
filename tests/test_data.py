"""Tests for data loading and preprocessing."""

import pytest
import torch
from PIL import Image

from adaptive_noise_schedule_diffusion_with_clip_guidance.data.preprocessing import (
    preprocess_image,
    preprocess_text,
    create_augmentation_transform,
    denormalize_image,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.data.loader import (
    ConceptualCaptionsDataset,
    get_dataloader,
)


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a test image
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))

        # Preprocess
        tensor = preprocess_image(img, size=64)

        # Check shape and range
        assert tensor.shape == (3, 64, 64)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_preprocess_image_conversion(self):
        """Test image mode conversion."""
        # Create grayscale image
        img = Image.new("L", (256, 256), color=128)

        # Should convert to RGB
        tensor = preprocess_image(img, size=64)

        assert tensor.shape == (3, 64, 64)

    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "  A  sample   caption  with   extra   spaces  "

        # Preprocess
        clean_text = preprocess_text(text)

        # Check cleaning
        assert clean_text == "A sample caption with extra spaces"

    def test_preprocess_text_truncation(self):
        """Test text truncation."""
        long_text = " ".join(["word"] * 100)

        # Truncate to 50 words
        truncated = preprocess_text(long_text, max_length=50)

        assert len(truncated.split()) == 50

    def test_create_augmentation_transform(self):
        """Test augmentation transform creation."""
        transform = create_augmentation_transform(size=64)

        # Create test image
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))

        # Apply transform
        tensor = transform(img)

        assert tensor.shape == (3, 64, 64)

    def test_denormalize_image(self):
        """Test image denormalization."""
        # Create normalized tensor
        tensor = torch.randn(3, 64, 64)

        # Denormalize
        denorm = denormalize_image(tensor)

        # Check range
        assert denorm.min() >= 0.0
        assert denorm.max() <= 1.0


class TestDataLoader:
    """Test data loading."""

    def test_conceptual_captions_dataset(self):
        """Test ConceptualCaptionsDataset."""
        # Create dataset with synthetic data
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=10,
            image_size=64,
            augment=False,
        )

        # Check length
        assert len(dataset) > 0

        # Get a sample
        sample = dataset[0]

        assert "image" in sample
        assert "caption" in sample
        assert sample["image"].shape == (3, 64, 64)
        assert isinstance(sample["caption"], str)

    def test_get_dataloader(self):
        """Test dataloader creation."""
        dataloader = get_dataloader(
            split="train",
            batch_size=2,
            max_samples=10,
            image_size=64,
            num_workers=0,
            shuffle=True,
        )

        # Get a batch
        batch = next(iter(dataloader))

        assert "image" in batch
        assert "caption" in batch
        assert batch["image"].shape[0] == 2  # Batch size
        assert len(batch["caption"]) == 2

    def test_dataloader_validation(self):
        """Test validation dataloader."""
        dataloader = get_dataloader(
            split="validation",
            batch_size=2,
            max_samples=5,
            image_size=64,
            num_workers=0,
            shuffle=False,
        )

        assert len(dataloader) > 0
