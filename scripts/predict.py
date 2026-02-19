#!/usr/bin/env python
"""Prediction script for generating images from text prompts."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
from PIL import Image
import numpy as np

from adaptive_noise_schedule_diffusion_with_clip_guidance.models.model import (
    AdaptiveNoiseDiffusionModel,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.utils.config import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of inference steps (overrides config)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)

    Returns:
        PIL Image
    """
    # Handle batch dimension
    if tensor.ndim == 4:
        tensor = tensor[0]

    # Convert to numpy
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    # Denormalize from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)

    # Convert to uint8
    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("Adaptive Noise Schedule Diffusion - Image Generation")
    logger.info("=" * 80)
    logger.info(f"Prompt: {args.prompt}")

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Set random seed
    set_seed(args.seed, deterministic=True)

    # Get device
    device = get_device()

    try:
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

        # Load checkpoint if exists
        checkpoint_path = Path(args.checkpoint)

        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Using untrained model")

        model.eval()

        # Determine number of inference steps
        if args.num_steps is not None:
            num_steps = args.num_steps
        else:
            if model.use_adaptive_schedule:
                num_steps = model_config.get("adaptive_inference_steps", 35)
            else:
                num_steps = model_config.get("num_inference_steps", 50)

        logger.info(f"Generating {args.num_images} image(s) with {num_steps} steps...")

        # Generate images
        prompts = [args.prompt] * args.num_images

        with torch.no_grad():
            images = model.generate(
                prompts=prompts,
                num_inference_steps=num_steps,
                guidance_scale=args.guidance_scale,
            )

        # Compute CLIP score
        clip_scores = model.get_clip_score(images, prompts)

        logger.info(f"Generated images with CLIP score: {clip_scores.mean().item():.4f}")

        # Save images
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.num_images == 1:
            # Save single image
            img = tensor_to_image(images[0])
            img.save(output_path)
            logger.info(f"Saved image to {output_path}")
        else:
            # Save multiple images
            for i, img_tensor in enumerate(images):
                img = tensor_to_image(img_tensor)
                img_path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
                img.save(img_path)
                logger.info(f"Saved image {i+1} to {img_path}")

        # Print summary
        logger.info("=" * 80)
        logger.info("Generation Summary:")
        logger.info(f"  Prompt: {args.prompt}")
        logger.info(f"  Images generated: {args.num_images}")
        logger.info(f"  Inference steps: {num_steps}")
        logger.info(f"  Guidance scale: {args.guidance_scale}")
        logger.info(f"  Mean CLIP score: {clip_scores.mean().item():.4f}")
        logger.info(f"  Output: {output_path}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Generation failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
