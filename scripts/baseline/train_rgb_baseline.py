import os
import sys
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/")
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models")
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/configs/baseline/")
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID")
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.dataset import SUNRGBDDataset
from models.baseline.ResNet18 import ResNet18

def train_baseline(cfg):
    data_root    = cfg.data.root
    toolbox_root = cfg.data.toolbox_root

    # Training Configuration
    epochs      = cfg.training.epochs
    batch_size  = cfg.training.batch_size
    lr          = cfg.training.lr
    num_workers = cfg.training.num_workers
    pretrained  = cfg.training.pretrained

    # RGB Image Configurations
    rgb_cfg     = cfg.modalities.rgb
    resize      = tuple(rgb_cfg.transforms.resize)
    mean        = rgb_cfg.transforms.normalize_mean
    std         = rgb_cfg.transforms.normalize_std
    ckpt_dir    = rgb_cfg.checkpoint_dir
    log_path    = rgb_cfg.logging.result_file

    # Point Cloud Configuration
    pc_cfg      = cfg.modalities.pointcloud
    num_points  = pc_cfg.num_points

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_rgb = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),])

    train_ds = SUNRGBDDataset(data_root=data_root,toolbox_root=toolbox_root,split='train', num_points=num_points, transform_rgb=transform_rgb,transform_depth=None, transform_points=None)

    val_ds = SUNRGBDDataset(
        data_root        = data_root,
        toolbox_root     = toolbox_root,
        split            = 'test',
        num_points       = num_points,
        transform_rgb    = transform_rgb,
        transform_depth  = None,
        transform_points = None
    )

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    # ░█▀▄▀█ █▀▀█ █▀▀▄ █▀▀ █── 
    # ░█░█░█ █──█ █──█ █▀▀ █── 
    # ░█──░█ ▀▀▀▀ ▀▀▀─ ▀▀▀ ▀▀▀

    model = ResNet18(num_classes=len(train_ds.class_to_idx),
                     pretrained=pretrained).to(device)


    # Training on PACE ICE
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # ─── prepare output dirs ─────────────────────────────────────────
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    best_val_acc = 0.0

    # ─── training loop ───────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = running_correct = running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for rgb, depth, pcl, labels in pbar:
            rgb, labels = rgb.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(rgb)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss    += loss.item() * bs
            preds            = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total   += bs

            pbar.set_postfix(
                loss=(running_loss / running_total),
                acc=(running_correct / running_total)
            )

        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total

        # ─── validation ───────────────────────────────────────────────
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for rgb, depth, pcl, labels in val_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                outputs = model(rgb)
                loss    = loss_fn(outputs, labels)

                bs = labels.size(0)
                val_loss    += loss.item() * bs
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total   += bs

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total

        # ─── checkpoint & log ─────────────────────────────────────────
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"baseline_epoch{epoch}.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_baseline.pth"))

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},"
                    f"{val_loss:.4f},{val_acc:.4f}\n")


if __name__ == "__main__":

    cfg_path = os.path.join(PROJECT_ROOT, "configs/baseline/rgb_config.yaml")

    cfg = OmegaConf.load(cfg_path)

    train_baseline(cfg)