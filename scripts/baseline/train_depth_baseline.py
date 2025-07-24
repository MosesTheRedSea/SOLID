import os
import sys
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/")
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/")
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/configs/baseline/")
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data")
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/scripts/")
import glob
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from constants import *
from dataset import SUNRGBDDataset
from baseline.ResNet18 import ResNet18
from torchvision.models import ResNet18_Weights

class DepthOnlyDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        _, depth, _, label = self.base[idx]
        return depth, label