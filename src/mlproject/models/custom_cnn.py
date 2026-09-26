"""
Family 1: Small Custom Convolutional Neural Network baseline (trained from scratch).
Mandated by ITRI626 Section 6: 'A small custom convolutional neural network as a baseline.'
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU + MaxPool block."""
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class CustomBaselineCNN(nn.Module):
    """
    Modular 4-stage convolutional baseline network.
    Feature Extractor: 3 -> 32 -> 64 -> 128 -> 256 channels.
    Pooling: Adaptive Average Pooling to handle variable input dimensions.
    Classifier: Multi-layer perceptron with dropout regularization.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),    # 224x224 -> 112x112
            ConvBlock(32, 64, pool=True),   # 112x112 -> 56x56
            ConvBlock(64, 128, pool=True),  # 56x56 -> 28x28
            ConvBlock(128, 256, pool=True), # 28x28 -> 14x14
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.global_pool(feats)
        logits = self.classifier(pooled)
        return logits
