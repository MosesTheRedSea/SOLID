import torch.nn as nn
from torchvision import models

"""
░█▀▀█ █▀▀ █▀▀ ░█▄─░█ █▀▀ ▀▀█▀▀ ── ▄█─ ▄▀▀▄ 
░█▄▄▀ █▀▀ ▀▀█ ░█░█░█ █▀▀ ──█── ▀▀ ─█─ ▄▀▀▄ 
░█─░█ ▀▀▀ ▀▀▀ ░█──▀█ ▀▀▀ ──▀── ── ▄█▄ ▀▄▄▀
"""

class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
