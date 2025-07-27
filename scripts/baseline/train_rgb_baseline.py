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
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from constants import *
from dataset2 import SUNRGBDDataset
from baseline.rgb.ResNet34 import ResNet34
from torchvision.models import ResNet34_Weights

class GrabRGBData(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
        self.class_to_idx = base_ds.class_to_idx
        self.classes = base_ds.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rgb, _, _, label, *_ = self.base[idx]
        return rgb, label

def train_baseline(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    num_workers = cfg.training.num_workers

    rgb_cfg = cfg.modalities.rgb
    ckpt_dir = rgb_cfg.checkpoint_dir
    log_path = rgb_cfg.logging.result_file

    resize = tuple(rgb_cfg.transforms.resize)
    mean, std = rgb_cfg.transforms.normalize_mean, rgb_cfg.transforms.normalize_std

    transform_rgb = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_train = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',  # or 'test'
        num_points=cfg.modalities.pointcloud.num_points,
        transform_rgb=transform_rgb,
        transform_depth=None,
        transform_points=None,
        allowed_classes=common_classes if 'base' in locals() else None,
        require_rgb=True,
        require_depth=False,
        require_points=False
    )

    full_val = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',  # or 'test'
        num_points=cfg.modalities.pointcloud.num_points,
        transform_rgb=transform_rgb,
        transform_depth=None,
        transform_points=None,
        allowed_classes=common_classes if 'base' in locals() else None,
        require_rgb=True,
        require_depth=False,
        require_points=False
    )

    common_classes = sorted(set(full_train.classes) & set(full_val.classes))
    print(f"Using {len(common_classes)} common classes between train and test")

    train_base = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',  # or 'test'
        num_points=cfg.modalities.pointcloud.num_points,
        transform_rgb=transform_rgb,
        transform_depth=None,
        transform_points=None,
        allowed_classes=common_classes if 'base' in locals() else None,
        require_rgb=True,
        require_depth=False,
        require_points=False
    )

    val_base = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',  # or 'test'
        num_points=cfg.modalities.pointcloud.num_points,
        transform_rgb=transform_rgb,
        transform_depth=None,
        transform_points=None,
        allowed_classes=common_classes if 'base' in locals() else None,
        require_rgb=True,
        require_depth=False,
        require_points=False
    )

    val_base.class_to_idx = train_base.class_to_idx
    val_base.classes = train_base.classes

    train_ds = GrabRGBData(train_base)
    val_ds = GrabRGBData(val_base)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_base.classes)
    weights = ResNet34_Weights.IMAGENET1K_V1
    model = ResNet34(num_classes=num_classes, weights=weights, dropout_prob=0.5, freeze_backbone=False).to(device)

    print("Final classification layer:", model.backbone.fc)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    ckpt_paths = sorted(
        [p for p in glob.glob(os.path.join(ckpt_dir, "rgb_baseline_*.pth")) if p.split("_")[-1].split(".")[0].isdigit()],
        key=lambda p: int(p.split("_")[-1].split(".")[0])
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

        for rgb, labels in tqdm(train_loader, desc="Train"):
            rgb, labels = rgb.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(rgb)
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
        print(f"  -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        val_loss = val_correct = val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for rgb, labels in tqdm(val_loader, desc="Validate"):
                rgb, labels = rgb.to(device), labels.to(device)
                out = model(rgb)
                loss = loss_fn(out, labels)

                preds = out.argmax(dim=1)
                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        val_loss /= val_total
        val_acc = val_correct / val_total
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        val_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        val_confusion = confusion_matrix(y_true, y_pred)

        print(f"  -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"     Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        scheduler.step(val_acc)

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
            'val_confusion_matrix': val_confusion.tolist()
        }

        torch.save(ckpt, os.path.join(ckpt_dir, f"rgb_baseline_{epoch}.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_rgb_baseline.pth"))

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f}\n")

if __name__ == "__main__":
    PROJECT_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID"
    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "configs/baseline/train_config.yaml"))
    train_baseline(cfg)
