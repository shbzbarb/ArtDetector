# models/resnet50_transfer.py
"""
ResNet-50 transfer-learning model + helpers.

- ResNet50TL: wraps torchvision ResNet-50, replaces the final FC with a 14-class head.
- freeze_backbone / unfreeze_all: utilities for 2-phase training.
- TemperatureScaler: post-hoc calibration module (single temperature T) with a
  version-safe .fit() that uses LBFGS on log_T to minimize NLL.

Compatible with torchvision >= 0.13 (uses Weights enums).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50TL(nn.Module):
    """
    ResNet-50 with a custom classification head.

    Args:
        num_classes: number of output classes.
        dropout: dropout probability before the final linear layer.
        pretrained: load ImageNet weights if True.
    """
    def __init__(self, num_classes: int, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        # Replace the classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def freeze_backbone(model: "ResNet50TL") -> None:
    """Freeze all layers except the custom classification head."""
    for name, p in model.backbone.named_parameters():
        # keep only the final head trainable (it's under "fc.*")
        if not name.startswith("fc."):
            p.requires_grad = False


def unfreeze_all(model: "ResNet50TL") -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for calibrated softmax confidence.

    Usage:
        scaler = TemperatureScaler(init_T=1.0).to(device)
        scaler.fit(val_logits, val_labels)   # optimizes log_T
        calibrated_logits = scaler(logits)   # logits / T
        calibrated_probs = softmax(calibrated_logits, dim=1)
    """
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        # optimize log_T for positivity and stability
        self.log_T = nn.Parameter(torch.log(torch.tensor(float(init_T))))

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    def fit(self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            max_iter: int = 50,
            lr: float = 0.01) -> None:
        """
        Fit temperature T on a held-out validation set by minimizing NLL.

        Args:
            logits: [N, C] validation logits (treated as constants).
            labels: [N] validation targets.
            max_iter: maximum LBFGS iterations.
            lr: LBFGS step size.

        Notes:
            - Do NOT wrap this method in no_grad; we need grads w.r.t. log_T only.
            - Only self.log_T has requires_grad=True.
        """
        if logits.numel() == 0 or labels.numel() == 0:
            # nothing to fit
            with torch.no_grad():
                self.log_T.data.zero_()  # T=1.0
            self.eval()
            return

        self.train()
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()


__all__ = [
    "ResNet50TL",
    "freeze_backbone",
    "unfreeze_all",
    "TemperatureScaler",
]