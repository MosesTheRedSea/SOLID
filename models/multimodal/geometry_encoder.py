import torch
import torch.nn as nn
import sys

sys.path.append('/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/baseline/cloud/dgcnn/pytorch/')
from model import DGCNN

class Args:
    def __init__(self):
        self.k = 20
        self.emb_dims = 512  # Match fusion dimension
        self.dropout = 0.3

class GeometryEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        args = Args()
        self.dgcnn = DGCNN(args=args, output_channels=out_dim)

    def forward(self, pcl):
        # pcl: (B, N, 6) → (B, 6, N)
        pcl = pcl.permute(0, 2, 1)
        return self.dgcnn(pcl)  # Output: (B, out_dim)