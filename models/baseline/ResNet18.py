import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

"""
░█▀▀█ █▀▀ █▀▀ ░█▄─░█ █▀▀ ▀▀█▀▀ ── ▄█─ ▄▀▀▄ 
░█▄▄▀ █▀▀ ▀▀█ ░█░█░█ █▀▀ ──█── ▀▀ ─█─ ▄▀▀▄ 
░█─░█ ▀▀▀ ▀▀▀ ░█──▀█ ▀▀▀ ──▀── ── ▄█▄ ▀▄▄▀
"""

class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, weights=None):
        super().__init__()
        if weights is not None:
            self.backbone = models.resnet18(weights=weights)
        else:
            w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=w)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
