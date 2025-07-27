import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score

sys.path.extend([
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/configs/baseline/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/scripts/"
])

from constants import *
from dataset2 import SUNRGBDDataset
from models.baseline.cloud.dgcnn.pytorch.model import DGCNN

def augment_pointcloud(pc):
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().numpy()  # Convert to NumPy if it's a Tensor

    theta = np.random.uniform(0, 2 * np.pi)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    pc[:, :3] = pc[:, :3] @ rot.T
    pc[:, :3] += 0.01 * np.random.randn(*pc[:, :3].shape)

    return torch.from_numpy(pc).float()

class GrabCloudData(Dataset):
    def __init__(self, base_ds):
        self.base = []
        for i in range(len(base_ds)):
            try:
                sample = base_ds[i]
                pcl = sample[2]
                if pcl is not None and pcl.shape[0] > 0:
                    pcl_aug = augment_pointcloud(pcl)
                    self.base.append((pcl_aug, sample[3]))
            except Exception as e:
                print(f"[WARN] Skipping sample {i} due to error: {e}")
        print(f"[INFO] GrabCloudData loaded {len(self.base)} point cloud samples.")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pcl, label = self.base[idx]
        xyz = pcl[:, :3]
        xyz = xyz - xyz.mean(dim=0)
        norm = torch.norm(xyz, dim=1).max()
        if norm > 0:
            xyz = xyz / norm
        return xyz.float(), label

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    xs = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x, _ in batch]
    ys = [y for _, y in batch]
    return torch.stack(xs), torch.tensor(ys)

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    num_points = cfg.modalities.pointcloud.num_points
    k = cfg.modalities.pointcloud.model.k
    dropout = cfg.modalities.pointcloud.model.dropout
    emb_dims = cfg.modalities.pointcloud.model.emb_dims
    lr = cfg.training.lr
    ckpt_dir = cfg.modalities.pointcloud.checkpoint_dir
    log_path = cfg.modalities.pointcloud.logging.result_file
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    train_full = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',
        num_points=num_points,
        transform_rgb=None,
        transform_depth=None,
        transform_points=lambda x: x,
        require_rgb=False,
        require_depth=False,
        require_points=True
    )

    val_full = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='test',
        num_points=num_points,
        transform_rgb=None,
        transform_depth=None,
        transform_points=lambda x: x,
        require_rgb=False,
        require_depth=False,
        require_points=True
    )

    common_classes = sorted(set(train_full.classes) & set(val_full.classes))
    print(f"[INFO] Using {len(common_classes)} common classes")

    train_base = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='train',
        num_points=num_points,
        transform_rgb=None,
        transform_depth=None,
        transform_points=lambda x: x,
        allowed_classes=common_classes,
        require_rgb=False,
        require_depth=False,
        require_points=True
    )

    val_base = SUNRGBDDataset(
        data_root=cfg.data.root,
        toolbox_root=cfg.data.toolbox_root,
        split='test',
        num_points=num_points,
        transform_rgb=None,
        transform_depth=None,
        transform_points=lambda x: x,
        allowed_classes=common_classes,
        require_rgb=False,
        require_depth=False,
        require_points=True
    )

    num_classes = len(common_classes)

    train_ds = GrabCloudData(train_base)
    val_ds = GrabCloudData(val_base)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate_fn)

    class Args: pass
    args = Args()
    args.emb_dims = emb_dims
    args.dropout = dropout
    args.k = k

    model = DGCNN(args, output_channels=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,precision,recall\n")

    ckpt_paths = sorted(
        [p for p in glob.glob(os.path.join(ckpt_dir, "dgcnn_baseline_*.pth")) if p.split("_")[-1].split(".")[0].isdigit()],
        key=lambda p: int(p.split("_")[-1].split(".")[0])
    )
    if ckpt_paths:
        latest = ckpt_paths[-1]
        print(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('val_accuracy', 0.0)
    else:
        print("No checkpoint found, starting from scratch")
        start_epoch = 1
        best_acc = 0.0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        preds_list, labels_list = [], []

        for xyz, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            if xyz is None:
                continue
            xyz = xyz.to(device).permute(0, 2, 1)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(xyz)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        print(f"[Train] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for xyz, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch}"):
                if xyz is None:
                    continue
                xyz = xyz.to(device).permute(0, 2, 1)
                labels = labels.to(device)

                out = model(xyz)
                loss = criterion(out, labels)
                preds = out.argmax(dim=1)

                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0) if val_total > 0 else 0.0
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0) if val_total > 0 else 0.0

        print(f"[Val] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

        scheduler.step(val_acc)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec
        }

        torch.save(ckpt, os.path.join(ckpt_dir, f"dgcnn_baseline_{epoch}.pth"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_dgcnn_baseline.pth"))

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_prec:.4f},{val_rec:.4f}\n")

if __name__ == "__main__":
    PROJECT_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID"
    cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "configs/baseline/train_config.yaml"))
    train(cfg)
