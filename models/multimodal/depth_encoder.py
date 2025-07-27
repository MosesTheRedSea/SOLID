import torch
import torch.nn as nn
from timm import create_model

class DepthEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            in_chans=1,
            features_only=False
        )
        self.fc = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        # Input: single-channel depth map (B, 1, 224, 224)
        features = self.backbone.forward_features(x)  # (B, num_features)
        return self.fc(features)                      # (B, out_dim)
