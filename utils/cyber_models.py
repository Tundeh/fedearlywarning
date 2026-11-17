"""Tabular models for network-flow classification.

The classifier is intentionally the final parameterized layer so the original
FedCCFA convention ``list(model.state_dict().keys())[-2:]`` still identifies
classifier weight and bias.
"""
from __future__ import annotations

import torch
from torch import nn


class CyberMLP(nn.Module):
    """Lightweight representation + classifier model for CIC flow features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = x.float()
        features = self.encoder(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
