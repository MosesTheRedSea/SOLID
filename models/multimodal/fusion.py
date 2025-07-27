import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=512, num_tokens=3):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, modality_feats):
        """
        modality_feats: list of tensors, each (B, D)
        Output: (B, 3 * D)
        """
        x = torch.stack(modality_feats, dim=1)  # (B, 3, D)
        x = x + self.pos_embed                  # (B, 3, D) with positional encoding
        x = self.transformer(x)                 # (B, 3, D)
        return x.flatten(start_dim=1)           # (B, 3*D)
