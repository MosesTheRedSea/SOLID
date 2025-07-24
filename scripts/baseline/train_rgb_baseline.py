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

class RGBOnlyDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        rgb, _, _, label, *_ = self.base[idx]
        return rgb, label

def train_baseline(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configs
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    num_workers = cfg.training.num_workers
    pretrained = cfg.training.pretrained

    # Paths
    rgb_cfg = cfg.modalities.rgb
    ckpt_dir = rgb_cfg.checkpoint_dir
    log_path = rgb_cfg.logging.result_file

    # Transforms
    resize = tuple(rgb_cfg.transforms.resize)
    mean, std = rgb_cfg.transforms.normalize_mean, rgb_cfg.transforms.normalize_std
    transform_rgb = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Base Datasets
    train_base = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',
        num_points=cfg.modalities.pointcloud.num_points,
        transform_rgb=transform_rgb,
        transform_depth=None, transform_points=None
    )
    val_base = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='test',
        num_points=cfg.modalities.pointcloud.num_points,
        transform_rgb=transform_rgb,
        transform_depth=None, transform_points=None
    )

    # Wrap to only RGB
    train_ds = RGBOnlyDataset(train_base)
    val_ds  = RGBOnlyDataset(val_base)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model setup
    num_classes = cfg.modalities.rgb.model.num_classes
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = ResNet18(num_classes=num_classes, pretrained=False, weights=weights).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    # Prepare output dirs
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Resume logic
    ckpt_paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "rgb_baseline_*.pth")),
        key=lambda p: int(os.path.splitext(p)[0].split("rgb_baseline_")[-1])
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

    # Log header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_precision,val_recall\n")

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        # -- Training --
        print(f"Epoch {epoch}/{epochs} - Training")
        model.train()
        running_loss = running_correct = running_total = 0
        for rgb, labels in tqdm(train_loader, desc="Train"):  
            rgb, labels = rgb.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(rgb)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss    += loss.item() * bs
            preds = out.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total   += bs

        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total
        print(f"  -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")

        # -- Validation --
        print(f"Epoch {epoch}/{epochs} - Validation")
        model.eval()
        val_loss = val_correct = val_total = 0
        all_preds, all_labels = [], []
        for rgb, labels in tqdm(val_loader, desc="Validate"):  
            rgb, labels = rgb.to(device), labels.to(device)
            out = model(rgb)
            loss = loss_fn(out, labels)

            bs = labels.size(0)
            val_loss    += loss.item() * bs
            preds = out.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += bs
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        val_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        val_confusion = confusion_matrix(y_true, y_pred)

        print(f"  -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"     Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print("  -> Confusion Matrix:")
        print(val_confusion, "\n")

        # -- Save Checkpoint --
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
            'val_confusion_matrix': val_confusion.tolist(),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"rgb_baseline_{epoch}.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_rgb_baseline.pth"))

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_acc:.4f},"
                f"{val_loss:.4f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f}\n"
            )

if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "configs/baseline/train_config.yaml"))

    train_baseline(cfg)
