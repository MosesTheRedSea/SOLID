import os
import sys

sys.path.extend([
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/configs/baseline/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/scripts/"
])

import glob
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from constants import *
from depthdata import SUNRGBDDataset
from baseline.depth.ResNet34 import ResNet34 
from torchvision.models import ResNet34_Weights

class GrabDepthData(Dataset):
    def __init__(self, base_ds, transform=None):
        self.transform = transform
        self.class_to_idx = base_ds.class_to_idx
        self.classes = base_ds.classes

        # Pre-filter valid indices
        self.valid_samples = []
        for i in range(len(base_ds)):
            try:
                depth_tensor, label = base_ds[i]
                if depth_tensor is None:
                    continue
                if isinstance(depth_tensor, torch.Tensor):
                    depth_np = depth_tensor.squeeze().cpu().numpy()
                else:
                    depth_np = depth_tensor
                if depth_np.ndim == 3:
                    depth_np = depth_np.squeeze()
                depth_np = (depth_np * 255.0).astype(np.uint8)
                _ = Image.fromarray(depth_np).convert("L")
                self.valid_samples.append(i)
            except Exception as e:
                print(f"[SKIP] Sample {i} invalid: {e}")
                continue
        self.base = base_ds
        print(f"[INFO] Using {len(self.valid_samples)} valid samples from {len(base_ds)} total")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        base_idx = self.valid_samples[idx]
        depth_tensor, label = self.base[base_idx]
        if isinstance(depth_tensor, torch.Tensor):
            depth_np = depth_tensor.squeeze().cpu().numpy()
        else:
            depth_np = depth_tensor
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze()
        depth_np = (depth_np * 255.0).astype(np.uint8)
        depth_img = Image.fromarray(depth_np).convert("L")
        if self.transform:
            depth_img = self.transform(depth_img)
        return depth_img, label

def train_depth(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs      = cfg.training.epochs
    batch_size  = cfg.training.batch_size
    lr          = cfg.training.lr
    num_workers = cfg.training.num_workers

    depth_cfg = cfg.modalities.depth
    ckpt_dir = depth_cfg.checkpoint_dir
    log_path = depth_cfg.logging.result_file

    resize = tuple(depth_cfg.transforms.resize)
    scale = depth_cfg.transforms.scale

    transform_depth = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * scale),
    ])

    # Align training and test classes
    full_train = SUNRGBDDataset(
        cfg.data.root, split='train',
        transform_depth=None
    )

    full_val = SUNRGBDDataset(
        cfg.data.root, split='test',
        transform_depth=None
    )

    common_classes = sorted(set(full_train.classes) & set(full_val.classes))
    print(f"Using {len(common_classes)} common classes between train and test")

    train_base = SUNRGBDDataset(
        cfg.data.root, split='train',
        transform_depth=None,
        allowed_classes=common_classes
    )

    val_base = SUNRGBDDataset(
        cfg.data.root, split='test',
        transform_depth=None,
        allowed_classes=common_classes
    )

    val_base.class_to_idx = train_base.class_to_idx
    val_base.classes = train_base.classes

    train_ds = GrabDepthData(train_base, transform=transform_depth)
    val_ds = GrabDepthData(val_base, transform=transform_depth)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_base.classes)
    model = ResNet34(
        num_classes=num_classes,
        in_channels=1,
        weights=ResNet34_Weights.IMAGENET1K_V1,
        dropout_prob=0.5,
        freeze_backbone=False
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    ckpt_paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "depth_baseline_*.pth")),
        key=lambda p: int(os.path.splitext(p)[0].split("depth_baseline_")[-1])
    )

    if ckpt_paths:
        latest = ckpt_paths[-1]
        print(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('val_accuracy', 0.0)
    else:
        print("No checkpoint found, starting from scratch")
        start_epoch = 1
        best_val_acc = 0.0

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_precision,val_recall\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = running_correct = running_total = 0

        for depth, labels in tqdm(train_loader, desc=f"Epoch {epoch} - Train"):
            depth, labels = depth.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(depth)
            loss = loss_fn(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            preds = out.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        print(f"[Train] Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        model.eval()
        val_loss = val_correct = val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for depth, labels in tqdm(val_loader, desc=f"Epoch {epoch} - Val"):
                depth, labels = depth.to(device), labels.to(device)
                out = model(depth)
                loss = loss_fn(out, labels)
                preds = out.argmax(dim=1)

                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        val_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        print(f"[Val] Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"[Val] Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(conf_matrix)

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_confusion_matrix': conf_matrix.tolist(),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"depth_baseline_{epoch}.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_depth_baseline.pth"))

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f}\n")


if __name__ == "__main__":
    PROJECT_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID"
    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "configs/baseline/train_config.yaml"))
    train_depth(cfg)
