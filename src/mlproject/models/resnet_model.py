"""
Family 2: Residual Network (ResNet) family with Transfer Learning.
Mandated by ITRI626 Section 6: 'ResNet, DenseNet, or MobileNetV3'.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class ResNetTransferModel(nn.Module):
    """
    ResNet-18 / ResNet-50 transfer learning wrapper.
    Leverages ImageNet pre-trained weights for residual feature representation.
    """
    def __init__(
        self,
        num_classes: int = 10,
        variant: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.variant = variant
        self.pretrained = pretrained

        if variant == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
        elif variant == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        # Optionally freeze backbone parameters for feature extraction
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
