#!/usr/bin/env python
"""Training script for adaptive noise schedule diffusion model."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch

from adaptive_noise_schedule_diffusion_with_clip_guidance.data.loader import get_dataloader
from adaptive_noise_schedule_diffusion_with_clip_guidance.models.model import (
    AdaptiveNoiseDiffusionModel,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.training.trainer import (
    DiffusionTrainer,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.utils.config import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.evaluation.analysis import (
    plot_training_curves,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train adaptive noise schedule diffusion model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("Adaptive Noise Schedule Diffusion Training")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Set random seed
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", True)
    set_seed(seed, deterministic)

    # Get device
    device = get_device()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize data loaders
        logger.info("Initializing data loaders...")

        data_config = config.get("data", {})
        training_config = config.get("training", {})

        train_loader = get_dataloader(
            split="train",
            batch_size=training_config.get("batch_size", 8),
            max_samples=data_config.get("max_samples_train", 10000),  # Reduced for faster training
            image_size=data_config.get("image_size", 512),
            num_workers=data_config.get("num_workers", 4),
            prefetch_factor=data_config.get("prefetch_factor", 2),
            shuffle=True,
        )

        val_loader = get_dataloader(
            split="validation",
            batch_size=training_config.get("batch_size", 8),
            max_samples=data_config.get("max_samples_val", 1000),
            image_size=data_config.get("image_size", 512),
            num_workers=data_config.get("num_workers", 4),
            prefetch_factor=data_config.get("prefetch_factor", 2),
            shuffle=False,
        )

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Initialize model
        logger.info("Initializing model...")

        model_config = config.get("model", {})

        model = AdaptiveNoiseDiffusionModel(
            diffusion_model_id=model_config.get("diffusion_model", "runwayml/stable-diffusion-v1-5"),
            clip_model_id=model_config.get("clip_model", "openai/clip-vit-base-patch32"),
            predictor_hidden_dim=model_config.get("predictor_hidden_dim", 256),
            predictor_num_layers=model_config.get("predictor_num_layers", 3),
            use_adaptive_schedule=model_config.get("use_adaptive_schedule", True),
            device=str(device),
        )

        logger.info("Model initialized successfully")

        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Checkpoint loaded successfully")

        # Initialize trainer
        logger.info("Initializing trainer...")

        trainer = DiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        # Train
        logger.info("Starting training...")

        history = trainer.train()

        logger.info("Training completed successfully")

        # Plot training curves
        logger.info("Generating training plots...")

        plot_training_curves(
            train_losses=history["train_losses"],
            val_losses=history["val_losses"],
            output_path=output_dir / "training_curves.png",
        )

        # Save final results
        import json

        results = {
            "best_val_loss": history["best_val_loss"],
            "training_time": history["training_time"],
            "config_path": args.config,
        }

        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {results_path}")
        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")
        logger.info(f"Training time: {history['training_time']:.2f} seconds")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
