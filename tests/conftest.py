"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "diffusion_model": "runwayml/stable-diffusion-v1-5",
            "clip_model": "openai/clip-vit-base-patch32",
            "predictor_hidden_dim": 128,
            "predictor_num_layers": 2,
            "use_adaptive_schedule": True,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "gradient_accumulation_steps": 1,
        },
        "data": {
            "image_size": 64,
            "max_samples_train": 10,
            "max_samples_val": 5,
        },
        "checkpoint": {
            "save_dir": "checkpoints",
            "save_every_n_epochs": 5,
            "keep_last_n": 3,
        },
        "early_stopping": {
            "patience": 10,
            "min_delta": 0.001,
        },
        "logging": {
            "use_mlflow": False,
        },
        "seed": 42,
    }


@pytest.fixture
def sample_images():
    """Generate sample images for testing."""
    # Create random images (2, 3, 64, 64)
    images = torch.randn(2, 3, 64, 64)
    # Normalize to [-1, 1]
    images = (images - images.min()) / (images.max() - images.min())
    images = (images - 0.5) * 2
    return images


@pytest.fixture
def sample_captions():
    """Sample text captions for testing."""
    return [
        "a cat sitting on a table",
        "a dog running in a park",
    ]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
