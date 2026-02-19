"""Custom loss functions and model components."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AdaptiveScheduleLoss(nn.Module):
    """
    Custom loss for training the adaptive noise schedule predictor.

    Combines reconstruction quality, semantic alignment, and efficiency objectives.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        semantic_weight: float = 0.5,
        efficiency_weight: float = 0.3,
    ):
        """
        Initialize adaptive schedule loss.

        Args:
            reconstruction_weight: Weight for image reconstruction loss
            semantic_weight: Weight for CLIP semantic alignment loss
            efficiency_weight: Weight for inference efficiency penalty
        """
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.semantic_weight = semantic_weight
        self.efficiency_weight = efficiency_weight

    def forward(
        self,
        predicted_image: torch.Tensor,
        target_image: torch.Tensor,
        clip_score: torch.Tensor,
        num_steps_used: torch.Tensor,
        max_steps: int = 50,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive schedule loss.

        Args:
            predicted_image: Generated image tensor
            target_image: Ground truth image tensor
            clip_score: CLIP similarity score between image and text
            num_steps_used: Number of denoising steps used
            max_steps: Maximum possible steps

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss (L2)
        recon_loss = F.mse_loss(predicted_image, target_image)

        # Semantic alignment loss (negative CLIP score - we want to maximize)
        semantic_loss = -clip_score.mean()

        # Efficiency penalty (encourage using fewer steps)
        # Normalized by max_steps to be in [0, 1] range
        step_ratio = num_steps_used.float() / max_steps
        efficiency_loss = step_ratio.mean()

        # Combined loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.semantic_weight * semantic_loss +
            self.efficiency_weight * efficiency_loss
        )

        loss_dict = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "semantic_loss": semantic_loss.item(),
            "efficiency_loss": efficiency_loss.item(),
            "avg_steps": num_steps_used.float().mean().item(),
        }

        return total_loss, loss_dict


class CLIPGuidedLoss(nn.Module):
    """
    CLIP-guided loss for semantic alignment during generation.

    Uses CLIP's image-text similarity to guide the diffusion process.
    """

    def __init__(
        self,
        guidance_scale: float = 150.0,
        target_score: float = 0.28,
    ):
        """
        Initialize CLIP-guided loss.

        Args:
            guidance_scale: Scale factor for CLIP guidance
            target_score: Target CLIP similarity score
        """
        super().__init__()
        self.guidance_scale = guidance_scale
        self.target_score = target_score

    def forward(
        self,
        clip_score: torch.Tensor,
        use_guidance: bool = True,
    ) -> torch.Tensor:
        """
        Compute CLIP-guided loss.

        Args:
            clip_score: CLIP similarity score
            use_guidance: Whether to apply guidance scaling

        Returns:
            CLIP guidance loss
        """
        if not use_guidance:
            return torch.tensor(0.0, device=clip_score.device)

        # Loss is negative scaled CLIP score (we want to maximize similarity)
        # Add target score offset to encourage reaching target
        loss = -self.guidance_scale * (clip_score - self.target_score).mean()

        return loss


class PreferenceRewardLoss(nn.Module):
    """
    Preference-based reward loss using RLHF principles.

    Learns from human preference data (UltraFeedback) to optimize generation quality.
    """

    def __init__(
        self,
        reward_scale: float = 0.1,
        kl_penalty: float = 0.02,
        margin: float = 0.5,
    ):
        """
        Initialize preference reward loss.

        Args:
            reward_scale: Scale factor for reward signal
            kl_penalty: KL divergence penalty coefficient
            margin: Margin for preference ranking loss
        """
        super().__init__()
        self.reward_scale = reward_scale
        self.kl_penalty = kl_penalty
        self.margin = margin

    def forward(
        self,
        preferred_score: torch.Tensor,
        rejected_score: torch.Tensor,
        kl_divergence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute preference-based reward loss.

        Args:
            preferred_score: Score for preferred generation
            rejected_score: Score for rejected generation
            kl_divergence: KL divergence from reference policy (optional)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Ranking loss with margin
        # We want: preferred_score > rejected_score + margin
        ranking_loss = F.relu(
            self.margin - (preferred_score - rejected_score)
        ).mean()

        # Scale by reward scale
        reward_loss = self.reward_scale * ranking_loss

        # Add KL penalty if provided
        total_loss = reward_loss
        kl_loss_value = 0.0

        if kl_divergence is not None:
            kl_loss = self.kl_penalty * kl_divergence.mean()
            total_loss = total_loss + kl_loss
            kl_loss_value = kl_loss.item()

        loss_dict = {
            "preference_loss": total_loss.item(),
            "ranking_loss": ranking_loss.item(),
            "kl_loss": kl_loss_value,
            "score_diff": (preferred_score - rejected_score).mean().item(),
        }

        return total_loss, loss_dict


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining features.

    Used in the noise schedule predictor to adaptively combine
    CLIP features and timestep embeddings.
    """

    def __init__(self, dim: int):
        """
        Initialize gated fusion layer.

        Args:
            dim: Feature dimension
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse two feature tensors with learned gating.

        Args:
            feature_a: First feature tensor
            feature_b: Second feature tensor

        Returns:
            Fused features
        """
        # Concatenate features
        concat = torch.cat([feature_a, feature_b], dim=-1)

        # Compute gate
        gate = self.gate(concat)

        # Gated fusion
        fused = gate * feature_a + (1 - gate) * feature_b

        return fused


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps.

    Converts scalar timesteps to high-dimensional embeddings.
    """

    def __init__(self, dim: int):
        """
        Initialize time embedding layer.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for timesteps.

        Args:
            timesteps: Timestep values (B,)

        Returns:
            Time embeddings (B, dim)
        """
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -embeddings
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings
