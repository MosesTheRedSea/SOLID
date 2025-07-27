import torch
import torch.nn as nn
from transformers import CLIPModel

class RGBEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        # Load pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = self.clip_model.vision_model
        self.projection = nn.Linear(self.vision_model.config.hidden_size, out_dim)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        vision_out = self.vision_model(pixel_values=x)
        pooled_output = vision_out.pooler_output  # (B, hidden_dim)
        return self.projection(pooled_output)     # (B, out_dim)
