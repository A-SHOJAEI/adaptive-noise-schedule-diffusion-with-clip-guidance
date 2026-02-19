"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: str,
    metrics: Optional[Dict[str, List[float]]] = None,
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Path to save plot
        metrics: Optional additional metrics to plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Additional metrics
    if metrics:
        for metric_name, metric_values in metrics.items():
            axes[1].plot(metric_values, label=metric_name, linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Training Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved training curves to {output_path}")


def plot_sample_generations(
    images: torch.Tensor,
    captions: List[str],
    output_path: str,
    num_samples: int = 9,
):
    """
    Plot sample generated images with captions.

    Args:
        images: Generated images tensor (N, 3, H, W)
        captions: List of text captions
        output_path: Path to save plot
        num_samples: Number of samples to plot
    """
    num_samples = min(num_samples, len(images))

    # Create grid
    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        row = i // ncols
        col = i % ncols

        # Get image
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))

        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

        # Plot
        axes[row, col].imshow(img)
        axes[row, col].set_title(captions[i][:50], fontsize=8)
        axes[row, col].axis('off')

    # Hide unused subplots
    for i in range(num_samples, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sample generations to {output_path}")


def plot_metric_comparison(
    adaptive_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    output_path: str,
):
    """
    Plot comparison of metrics between adaptive and baseline models.

    Args:
        adaptive_metrics: Metrics for adaptive model
        baseline_metrics: Metrics for baseline model
        output_path: Path to save plot
    """
    # Extract common metrics
    metric_names = list(set(adaptive_metrics.keys()) & set(baseline_metrics.keys()))

    if not metric_names:
        logger.warning("No common metrics to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metric_names))
    width = 0.35

    adaptive_values = [adaptive_metrics[m] for m in metric_names]
    baseline_values = [baseline_metrics[m] for m in metric_names]

    ax.bar(x - width/2, adaptive_values, width, label='Adaptive', alpha=0.8)
    ax.bar(x + width/2, baseline_values, width, label='Baseline', alpha=0.8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved metric comparison to {output_path}")


def generate_evaluation_report(
    metrics: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate comprehensive evaluation report.

    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save report

    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {json_path}")

    # Generate text report
    report_path = output_dir / "evaluation_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary metrics
        f.write("Key Metrics:\n")
        f.write("-" * 80 + "\n")

        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                f.write(f"{key:30s}: {value:>10.4f}\n")
            else:
                f.write(f"{key:30s}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"Saved evaluation report to {report_path}")

    return str(report_path)


def save_samples_as_images(
    images: torch.Tensor,
    captions: List[str],
    output_dir: str,
    prefix: str = "sample",
):
    """
    Save individual generated images to files.

    Args:
        images: Generated images tensor
        captions: Text captions
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, caption) in enumerate(zip(images, captions)):
        # Convert to PIL
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2  # Denormalize
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img)

        # Save
        filename = f"{prefix}_{i:04d}.png"
        pil_img.save(output_dir / filename)

    logger.info(f"Saved {len(images)} images to {output_dir}")


def create_results_table(metrics: Dict[str, float]) -> str:
    """
    Create a formatted results table for README.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Markdown formatted table string
    """
    table = "| Metric | Value |\n"
    table += "|--------|-------|\n"

    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            table += f"| {key} | {value:.4f} |\n"
        else:
            table += f"| {key} | {value} |\n"

    return table
