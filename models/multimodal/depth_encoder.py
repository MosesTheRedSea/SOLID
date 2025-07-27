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
            num_classes=0  # disable classification head
        )
        self.fc = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        features = self.backbone.forward_features(x)  # Shape: [B, H, W, C] -> [8, 7, 7, 768]
        # The pooling logic needs to be corrected for channels-last format
        if features.dim() == 4:
            # FAULTY LINE:
            # features = features.mean(dim=[2, 3])
            # CORRECTED LINE: Average over H and W dims (1 and 2)
            features = features.mean(dim=[1, 2])
        elif features.dim() == 3:  # This handles (B, L, C) format, which is fine
            features = features.mean(dim=1)
        return self.fc(features)