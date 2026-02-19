"""Configuration utilities."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved configuration to {output_path}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
    )

    logger.info(f"Logging initialized at {log_level} level")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {seed} (deterministic mode)")
    else:
        torch.backends.cudnn.benchmark = True
        logger.info(f"Set random seed to {seed} (non-deterministic mode)")


def get_device() -> torch.device:
    """
    Get the best available device.

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on

    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {path}")

    return checkpoint
