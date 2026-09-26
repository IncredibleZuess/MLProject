"""
Family 3: Modern ConvNet / Vision Transformer Architecture.
Mandated by ITRI626 Section 6: 'EfficientNetV2 or ConvNeXt / Vision Transformer or DeiT'.
Provides support for Vision Transformer (ViT-B/16) and MobileNetV3 / EfficientNetV2.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ViT_B_16_Weights,
    EfficientNet_V2_S_Weights,
    mobilenet_v3_large,
    vit_b_16,
    efficientnet_v2_s,
)


class MobileNetOrViTModel(nn.Module):
    """
    Modern Architecture Family:
    - 'vit_b_16': Vision Transformer with self-attention mechanism
    - 'mobilenet_v3': Efficient Inverted Residual blocks with squeeze-and-excitation
    - 'efficientnet_v2_s': Fused inverted bottleneck convolution architecture
    """
    def __init__(
        self,
        num_classes: int = 10,
        variant: str = "mobilenet_v3",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.variant = variant
        self.pretrained = pretrained

        if variant == "mobilenet_v3":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = mobilenet_v3_large(weights=weights)
            in_features = self.model.classifier[0].out_features
            # Replace final classification head
            self.model.classifier[3] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        elif variant == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.model = efficientnet_v2_s(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        elif variant == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.model = vit_b_16(weights=weights)
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unsupported modern architecture variant: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
