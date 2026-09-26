"""
Configuration module for the Microcontroller & Electronic Component Classification Project.
Adheres to ITRI626 practical project specifications.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import os
import torch


@dataclass
class ProjectConfig:
    # Reproducibility
    seed: int = 42
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "processed")
    splits_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "splits")
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs" / "models")
    figures_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs" / "figures")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "outputs" / "results")

    # Image properties
    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    # Standard ImageNet normalization statistics
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Dataset split ratios (70 / 15 / 15 as recommended in ITRI626 spec)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Target classes for Microcontroller & Electronic Component Classification
    target_classes: List[str] = field(default_factory=lambda: [
        "arduino_uno",
        "arduino_nano",
        "esp32",
        "esp8266",
        "raspberry_pi_pico",
        "stm32_bluepill",
        "microcontroller_ic",
        "resistor",
        "capacitor",
        "diode_led",
    ])

    # Training Hyperparameters
    batch_size: int = 32
    # On Windows, PyTorch multiprocessing requires num_workers=0 to avoid spawn crashes
    num_workers: int = 0 if os.name == "nt" else 2
    epochs: int = 25
    early_stopping_patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Three model families required by ITRI626
    # 1. Custom CNN baseline (from scratch)
    # 2. ResNet family (pretrained transfer learning)
    # 3. Vision Transformer / Modern ConvNet family (pretrained transfer learning)
    model_families: List[str] = field(default_factory=lambda: [
        "custom_cnn",
        "resnet18",
        "mobilenet_v3",
    ])

    def get_device(self) -> torch.device:
        """Detect best available device (CUDA GPU or CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def ensure_directories(self) -> None:
        """Create required directories if they don't already exist."""
        for directory in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.splits_dir,
            self.outputs_dir,
            self.models_dir,
            self.figures_dir,
            self.results_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


default_config = ProjectConfig()
