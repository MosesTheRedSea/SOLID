import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_dim=1536, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
