"""Core model implementations for adaptive noise schedule diffusion."""

import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

from adaptive_noise_schedule_diffusion_with_clip_guidance.models.components import (
    GatedFusion,
    TimeEmbedding,
)

logger = logging.getLogger(__name__)


class NoiseSchedulePredictor(nn.Module):
    """
    Lightweight predictor network for adaptive noise scheduling.

    Predicts optimal noise levels per timestep based on CLIP features
    and current generation state.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        time_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_timesteps: int = 1000,
    ):
        """
        Initialize noise schedule predictor.

        Args:
            clip_dim: CLIP feature dimension
            time_dim: Time embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            max_timesteps: Maximum diffusion timesteps
        """
        super().__init__()

        self.clip_dim = clip_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps

        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)

        # Feature fusion
        self.fusion = GatedFusion(hidden_dim)

        # CLIP feature projection
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)

        # Time projection
        self.time_proj = nn.Linear(time_dim, hidden_dim)

        # MLP layers for prediction
        layers = []
        input_dim = hidden_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output heads
        self.skip_head = nn.Linear(hidden_dim, 1)  # Predict whether to skip this step
        self.noise_head = nn.Linear(hidden_dim, 1)  # Predict noise level adjustment

    def forward(
        self,
        clip_features: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict adaptive noise schedule.

        Args:
            clip_features: CLIP image-text alignment features (B, clip_dim)
            timestep: Current diffusion timestep (B,)

        Returns:
            Dictionary containing 'skip_prob' and 'noise_scale'
        """
        # Embed timestep
        time_emb = self.time_embed(timestep)  # (B, time_dim)

        # Project features
        clip_feat = self.clip_proj(clip_features)  # (B, hidden_dim)
        time_feat = self.time_proj(time_emb)  # (B, hidden_dim)

        # Fuse features
        fused = self.fusion(clip_feat, time_feat)  # (B, hidden_dim)

        # MLP processing
        hidden = self.mlp(fused)  # (B, hidden_dim)

        # Predictions
        skip_logit = self.skip_head(hidden)  # (B, 1)
        noise_scale = self.noise_head(hidden)  # (B, 1)

        return {
            "skip_prob": torch.sigmoid(skip_logit.squeeze(-1)),  # (B,)
            "noise_scale": torch.sigmoid(noise_scale.squeeze(-1)),  # (B,)
        }


class AdaptiveNoiseDiffusionModel(nn.Module):
    """
    Text-to-image diffusion model with adaptive noise scheduling.

    Combines Stable Diffusion with a learned predictor for dynamic
    noise schedule adjustment based on CLIP-measured semantic alignment.
    """

    def __init__(
        self,
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        predictor_hidden_dim: int = 256,
        predictor_num_layers: int = 3,
        use_adaptive_schedule: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize adaptive noise diffusion model.

        Args:
            diffusion_model_id: HuggingFace model ID for diffusion model
            clip_model_id: HuggingFace model ID for CLIP
            predictor_hidden_dim: Hidden dimension for schedule predictor
            predictor_num_layers: Number of layers in schedule predictor
            use_adaptive_schedule: Whether to use adaptive scheduling
            device: Device to load models on
        """
        super().__init__()

        self.device = device
        self.use_adaptive_schedule = use_adaptive_schedule

        logger.info(f"Loading diffusion model: {diffusion_model_id}")

        try:
            # Load Stable Diffusion pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                diffusion_model_id,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.pipe = self.pipe.to(device)

            # Extract components
            self.unet = self.pipe.unet
            self.vae = self.pipe.vae
            self.text_encoder = self.pipe.text_encoder
            self.tokenizer = self.pipe.tokenizer
            self.scheduler = self.pipe.scheduler

        except Exception as e:
            logger.warning(f"Failed to load full pipeline: {e}")
            logger.info("Using lightweight model components")
            self._initialize_lightweight_components(device)

        # Load CLIP model
        logger.info(f"Loading CLIP model: {clip_model_id}")
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

        # Initialize adaptive schedule predictor
        if use_adaptive_schedule:
            logger.info("Initializing adaptive schedule predictor")
            self.schedule_predictor = NoiseSchedulePredictor(
                clip_dim=512,  # CLIP feature dimension
                hidden_dim=predictor_hidden_dim,
                num_layers=predictor_num_layers,
            ).to(device)
        else:
            self.schedule_predictor = None
            logger.info("Using fixed noise schedule")

    def _initialize_lightweight_components(self, device: str):
        """Initialize lightweight model components for testing."""
        from diffusers import AutoencoderKL

        # Minimal UNet
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=768,
        ).to(device)

        # VAE for latent encoding/decoding
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(device)

        # Text encoder and tokenizer
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Scheduler
        self.scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

    def encode_prompt(self, prompt: List[str]) -> torch.Tensor:
        """
        Encode text prompts to embeddings.

        Args:
            prompt: List of text prompts

        Returns:
            Text embeddings tensor
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]

        return text_embeddings

    def get_clip_score(
        self,
        images: torch.Tensor,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Compute CLIP similarity score between images and texts.

        Args:
            images: Image tensors (B, C, H, W)
            texts: List of text captions

        Returns:
            CLIP similarity scores (B,)
        """
        with torch.no_grad():
            # Process inputs
            inputs = self.clip_processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get CLIP features
            outputs = self.clip_model(**inputs)

            # Compute similarity
            similarity = outputs.logits_per_image.diagonal()

            # Normalize to [0, 1]
            similarity = similarity / 100.0

        return similarity

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate images from text prompts.

        Args:
            prompts: List of text prompts
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            **kwargs: Additional generation parameters

        Returns:
            Generated images tensor
        """
        if hasattr(self, 'pipe'):
            # Use full pipeline
            output = self.pipe(
                prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs,
            )
            return output.images
        else:
            # Simplified generation
            batch_size = len(prompts)

            # Encode prompts
            text_embeddings = self.encode_prompt(prompts)

            # Random latents
            latents = torch.randn(
                batch_size,
                4,
                64,
                64,
                device=self.device,
                dtype=text_embeddings.dtype,
            )

            # Denoise
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.scheduler.timesteps:
                # Predict noise
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

                # Update latents
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode to images
            images = self.vae.decode(latents / 0.18215).sample

            return images
