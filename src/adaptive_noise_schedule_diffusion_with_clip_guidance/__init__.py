"""
Adaptive Noise Schedule Diffusion with CLIP Guidance.

A text-to-image diffusion model with dynamically adaptive noise scheduling
that adjusts denoising steps based on CLIP-measured semantic alignment.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_noise_schedule_diffusion_with_clip_guidance.models.model import (
    AdaptiveNoiseDiffusionModel,
    NoiseSchedulePredictor,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.training.trainer import (
    DiffusionTrainer,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.evaluation.metrics import (
    compute_fid_score,
    compute_clip_score,
    compute_inference_speedup,
)

__all__ = [
    "AdaptiveNoiseDiffusionModel",
    "NoiseSchedulePredictor",
    "DiffusionTrainer",
    "compute_fid_score",
    "compute_clip_score",
    "compute_inference_speedup",
]
