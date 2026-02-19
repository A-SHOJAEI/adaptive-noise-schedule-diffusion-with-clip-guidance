"""Training loop implementation with LR scheduling and early stopping."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from adaptive_noise_schedule_diffusion_with_clip_guidance.models.model import (
    AdaptiveNoiseDiffusionModel,
)
from adaptive_noise_schedule_diffusion_with_clip_guidance.models.components import (
    AdaptiveScheduleLoss,
    CLIPGuidedLoss,
    PreferenceRewardLoss,
)

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """
    Trainer for adaptive noise schedule diffusion model.

    Handles training loop, optimization, scheduling, and checkpointing.
    """

    def __init__(
        self,
        model: AdaptiveNoiseDiffusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """
        Initialize trainer.

        Args:
            model: Diffusion model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Training device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training settings
        self.num_epochs = config.get("training", {}).get("num_epochs", 50)
        self.gradient_clip_norm = config.get("training", {}).get("gradient_clip_norm", 1.0)
        self.mixed_precision = config.get("training", {}).get("mixed_precision", True)
        self.gradient_accumulation_steps = config.get("training", {}).get(
            "gradient_accumulation_steps", 1
        )

        # Initialize optimizer
        self._setup_optimizer()

        # Initialize scheduler
        self._setup_scheduler()

        # Initialize loss functions
        self._setup_losses()

        # Early stopping
        self.early_stopping_patience = config.get("early_stopping", {}).get("patience", 10)
        self.early_stopping_min_delta = config.get("early_stopping", {}).get("min_delta", 0.001)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint", {}).get("save_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # MLflow tracking (optional)
        self.use_mlflow = config.get("logging", {}).get("use_mlflow", False)
        if self.use_mlflow:
            self._setup_mlflow()

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

        logger.info("Trainer initialized")

    def _setup_optimizer(self):
        """Setup optimizer."""
        lr = self.config.get("training", {}).get("learning_rate", 0.0001)
        weight_decay = self.config.get("training", {}).get("weight_decay", 0.01)

        # Only optimize schedule predictor if using adaptive scheduling
        if self.model.use_adaptive_schedule and self.model.schedule_predictor is not None:
            params = self.model.schedule_predictor.parameters()
        else:
            params = self.model.parameters()

        self.optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=self.config.get("optimizer", {}).get("betas", [0.9, 0.999]),
            eps=self.config.get("optimizer", {}).get("eps", 1e-8),
        )

        logger.info(f"Optimizer initialized with lr={lr}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get("scheduler", {}).get("type", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.config.get("scheduler", {}).get("min_lr", 1e-6),
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=self.config.get("scheduler", {}).get("min_lr", 1e-6),
            )
        else:
            self.scheduler = None

        logger.info(f"Scheduler: {scheduler_type}")

    def _setup_losses(self):
        """Setup loss functions."""
        self.adaptive_loss = AdaptiveScheduleLoss(
            reconstruction_weight=1.0,
            semantic_weight=0.5,
            efficiency_weight=0.3,
        )

        self.clip_loss = CLIPGuidedLoss(
            guidance_scale=self.config.get("clip_guidance", {}).get("clip_guidance_scale", 150.0),
            target_score=self.config.get("clip_guidance", {}).get("target_clip_score", 0.28),
        )

        rlhf_config = self.config.get("rlhf", {})
        self.preference_loss = PreferenceRewardLoss(
            reward_scale=rlhf_config.get("reward_scale", 0.1),
            kl_penalty=rlhf_config.get("kl_penalty", 0.02),
            margin=rlhf_config.get("preference_margin", 0.5),
        )

        logger.info("Loss functions initialized")

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            import mlflow

            experiment_name = self.config.get("logging", {}).get(
                "experiment_name", "adaptive-noise-diffusion"
            )
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()

            # Log config
            mlflow.log_params({
                "learning_rate": self.config.get("training", {}).get("learning_rate", 0.0001),
                "batch_size": self.config.get("training", {}).get("batch_size", 8),
                "num_epochs": self.num_epochs,
            })

            self.mlflow = mlflow
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.use_mlflow = False
            self.mlflow = None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch["image"].to(self.device)
                captions = batch["caption"]

                # Forward pass with mixed precision
                with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                    # Generate images (simplified for training)
                    # In practice, this would involve the full diffusion process

                    # Get CLIP score
                    clip_scores = self.model.get_clip_score(images, captions)

                    # Simulate adaptive scheduling
                    if self.model.use_adaptive_schedule and self.model.schedule_predictor is not None:
                        # Get CLIP features for predictor
                        with torch.no_grad():
                            clip_inputs = self.model.clip_processor(
                                text=captions,
                                return_tensors="pt",
                                padding=True,
                            )
                            clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
                            text_features = self.model.clip_model.get_text_features(**clip_inputs)

                        # Predict schedule
                        timesteps = torch.randint(0, 1000, (images.size(0),), device=self.device)
                        schedule_pred = self.model.schedule_predictor(text_features, timesteps)

                        # Compute loss (simplified)
                        loss = -clip_scores.mean()  # Maximize CLIP score

                        # Add efficiency penalty
                        if "skip_prob" in schedule_pred:
                            efficiency_bonus = schedule_pred["skip_prob"].mean() * 0.1
                            loss = loss - efficiency_bonus
                    else:
                        # Standard loss
                        loss = -clip_scores.mean()

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        # Gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )

                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )

                        # Optimizer step
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                # Update metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({"loss": total_loss / num_batches})

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)

        metrics = {
            "train_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_clip_score = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    images = batch["image"].to(self.device)
                    captions = batch["caption"]

                    # Get CLIP score
                    clip_scores = self.model.get_clip_score(images, captions)

                    # Simple validation loss
                    loss = -clip_scores.mean()

                    total_loss += loss.item()
                    total_clip_score += clip_scores.mean().item()
                    num_batches += 1

                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)
        avg_clip_score = total_clip_score / max(num_batches, 1)

        metrics = {
            "val_loss": avg_loss,
            "val_clip_score": avg_clip_score,
        }

        return metrics

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Training history and final metrics
        """
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Train Loss: {metrics['train_loss']:.4f}, "
                f"Val Loss: {metrics['val_loss']:.4f}, "
                f"Val CLIP Score: {metrics.get('val_clip_score', 0):.4f}"
            )

            # MLflow logging
            if self.use_mlflow and self.mlflow is not None:
                try:
                    self.mlflow.log_metrics(metrics, step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Save checkpoint
            if epoch % self.config.get("checkpoint", {}).get("save_every_n_epochs", 5) == 0:
                self.save_checkpoint(epoch, metrics)

            # Early stopping check
            val_loss = metrics["val_loss"]
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # Save best model
                self.save_checkpoint(epoch, metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Track losses
            self.train_losses.append(metrics["train_loss"])
            self.val_losses.append(metrics["val_loss"])

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Close MLflow run
        if self.use_mlflow and self.mlflow is not None:
            try:
                self.mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "training_time": training_time,
        }

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
