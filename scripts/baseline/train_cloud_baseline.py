import os
import sys

sys.path.extend([
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/configs/baseline/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/scripts/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/baseline/cloud"])

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
from models.baseline.cloud.dgcnn import DGCNN
from models.baseline.cloud.dgcnn.util import cal_loss

class GrabCloudData(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _, _, pcl, label, *_ = self.base[idx]
        xyz = pcl[:, :3] 
        xyz = xyz - xyz.mean(axis=0)  
        norm = np.max(np.linalg.norm(xyz, axis=1))
        xyz = xyz / norm
        return xyz.astype('float32'), label

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load data ===
    base_train = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',
        num_points=cfg.modalities.pointcloud.num_points,
    )
    base_test = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='test',
        num_points=cfg.modalities.pointcloud.num_points,
    )

    train_ds = GrabCloudData(base_train)
    test_ds  = GrabCloudData(base_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    # === Model ===
    model = DGCNN(k=cfg.model.k, emb_dims=cfg.model.emb_dims,
                  dropout=cfg.model.dropout, output_channels=len(base_train.classes)).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs, eta_min=cfg.training.lr)
    criterion = cal_loss

    best_val_acc = 0.0

    for epoch in range(1, cfg.training.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.training.epochs}")
        model.train()
        
        running_loss = running_correct = running_total = 0
        for xyz, labels in tqdm(train_loader, desc="Train"):
            xyz, labels = xyz.to(device), labels.to(device).squeeze()
            xyz = xyz.permute(0, 2, 1)
            optimizer.zero_grad()
            out = model(xyz)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        print(f"[Train] Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xyz, labels in tqdm(test_loader, desc="Val"):
                xyz, labels = xyz.to(device), labels.to(device).squeeze()
                xyz = xyz.permute(0, 2, 1)
                out = model(xyz)
                loss = criterion(out, labels)
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
        torch.save(ckpt, os.path.join(cfg.model.ckpt_dir, f"dgcnn_epoch_{epoch}.pth"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cfg.model.ckpt_dir, "best_dgcnn.pth"))

        with open(cfg.model.log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f}\n")

if __name__ == "__main__":
    config_path = os.path.join(PROJECT_ROOT, "configs/baseline/train_config.yaml")
    cfg = OmegaConf.load(config_path)
    train(cfg)