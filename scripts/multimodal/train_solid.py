import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import glob

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight

sys.path.extend([
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/multimodal/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/"
])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from solid_pipeline import SolidFusionPipeline
from dataset import SUNRGBDDataset

SUNRGBD_DATA_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBD"
SUNRGBD_TOOLBOX_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBDtoolbox"
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/outputs/checkpoints/multimodal"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_fusion_model.pt")
RESULTS_DIR = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/outputs/results/multimodal"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DepthTransform:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.mean = 3000.0
        self.std = 1500.0

    def __call__(self, depth_map):
        depth_tensor = torch.from_numpy(depth_map.copy()).float().unsqueeze(0)
        resized_tensor = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0), size=self.size, mode='nearest'
        ).squeeze(0)
        normalize = transforms.Normalize(mean=[self.mean], std=[self.std])
        return normalize(resized_tensor)

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_depth = DepthTransform()

train_dataset = SUNRGBDDataset(
    data_root=SUNRGBD_DATA_ROOT,
    toolbox_root=SUNRGBD_TOOLBOX_ROOT,
    split='train',
    num_points=1024,
    transform_rgb=transform_rgb,
    transform_depth=transform_depth
)
NUM_CLASSES = len(train_dataset.classes)

test_dataset = SUNRGBDDataset(
    data_root=SUNRGBD_DATA_ROOT,
    toolbox_root=SUNRGBD_TOOLBOX_ROOT,
    split='test',
    num_points=1024,
    transform_rgb=transform_rgb,
    transform_depth=transform_depth,
    allowed_classes=train_dataset.classes
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = SolidFusionPipeline(num_classes=NUM_CLASSES).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
start_epoch = 0
best_acc = 0.0

ckpt_paths = sorted(
    glob.glob(os.path.join(CHECKPOINT_DIR, "fusion_model_epoch_*.pt")),
    key=lambda p: int(p.split("_")[-1].split(".")[0])
)

if ckpt_paths:
    latest_ckpt_path = ckpt_paths[-1]
    print(f"Resuming training from checkpoint: {latest_ckpt_path}")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint.get('best_accuracy', 0.0)
    print(f"Resumed from epoch {start_epoch}. Best accuracy so far: {best_acc:.4f}")
else:
    print("No checkpoint found. Starting training from scratch.")

for epoch in range(start_epoch, EPOCHS):
    
    model.train()
    train_losses = []
    all_preds_train = []
    all_labels_train = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for rgb, depth, pcl, labels, _ in progress_bar:
        rgb, depth, pcl, labels = rgb.to(device), depth.to(device), pcl.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(rgb, depth, pcl)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        all_preds_train.extend(preds.cpu().numpy())
        all_labels_train.extend(labels.cpu().numpy())
        progress_bar.set_postfix(loss=np.mean(train_losses))

    train_acc = accuracy_score(all_labels_train, all_preds_train)
    train_f1 = f1_score(all_labels_train, all_preds_train, average='macro', zero_division=0)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {np.mean(train_losses):.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

    model.eval()
    all_preds_test = []
    all_labels_test = []

    with torch.no_grad():
        for rgb, depth, pcl, labels, _ in test_loader:
            rgb, depth, pcl = rgb.to(device), depth.to(device), pcl.to(device)
            outputs = model(rgb, depth, pcl)
            preds = torch.argmax(outputs, dim=1)
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.numpy())

    test_acc = accuracy_score(all_labels_test, all_preds_test)
    test_precision = precision_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
    test_recall = recall_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels_test, all_preds_test, average='macro', zero_division=0)
    test_cm = confusion_matrix(all_labels_test, all_preds_test)

    print(f"Test Metrics | Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

    is_best = test_acc > best_acc
    if is_best:
        best_acc = test_acc

    epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"fusion_model_epoch_{epoch+1}.pt")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'metrics': {
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_macro': test_f1,
            'confusion_matrix': test_cm
        }
    }, epoch_ckpt_path)

    if is_best:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved at {BEST_MODEL_PATH} with accuracy: {best_acc:.4f}")
    
    log_file = os.path.join(RESULTS_DIR, "solid_results.txt")
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch+1}:\n")
        f.write(f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, "
                f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}\n")
        f.write("-" * 50 + "\n")

print("\nTraining complete.")