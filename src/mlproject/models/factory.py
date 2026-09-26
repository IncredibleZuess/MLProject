"""
Factory for instantiating models and extracting parameter counts / model metadata.
"""

from typing import Dict, Any
import torch
import torch.nn as nn

from mlproject.models.custom_cnn import CustomBaselineCNN
from mlproject.models.resnet_model import ResNetTransferModel
from mlproject.models.transformer_model import MobileNetOrViTModel


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Builds one of the three required model families:
    1. 'custom_cnn': Baseline Custom CNN (from scratch)
    2. 'resnet18' or 'resnet50': ResNet Family (transfer learning)
    3. 'mobilenet_v3', 'efficientnet_v2_s', or 'vit_b_16': Modern / Vision Transformer family
    """
    model_name = model_name.lower().strip()
    if model_name in ["custom_cnn", "baseline_cnn"]:
        return CustomBaselineCNN(num_classes=num_classes, dropout=dropout)
    elif model_name in ["resnet18", "resnet50"]:
        return ResNetTransferModel(num_classes=num_classes, variant=model_name, pretrained=pretrained, dropout=dropout)
    elif model_name in ["mobilenet_v3", "efficientnet_v2_s", "vit_b_16"]:
        return MobileNetOrViTModel(num_classes=num_classes, variant=model_name, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: 'custom_cnn', 'resnet18', 'resnet50', 'mobilenet_v3', 'efficientnet_v2_s', 'vit_b_16'."
        )


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Calculate parameter counts and estimated model size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4 bytes per float32 parameter
    param_size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(param_size_mb, 2),
    }
