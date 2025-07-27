import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torchvision import transforms

sys.path.extend([
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/multimodal/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/"
])

from solid_pipeline import SolidFusionPipeline
from dataset import SUNRGBDDataset

SUNRGBD_DATA_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBD"
SUNRGBD_TOOLBOX_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBDtoolbox"
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "outputs/checkpoints/multimodal/fusion_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_depth = transforms.Compose([
    lambda x: transforms.functional.resize(transforms.functional.to_pil_image(x), (224, 224)),
    transforms.ToTensor()
])

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = SolidFusionPipeline(num_classes=NUM_CLASSES).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    all_preds = []
    all_labels = []

    for rgb, depth, pcl, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        rgb, depth, pcl, labels = rgb.to(device), depth.to(device), pcl.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb, depth, pcl)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Train Loss: {np.mean(train_losses):.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for rgb, depth, pcl, labels, _ in test_loader:
            rgb, depth, pcl = rgb.to(device), depth.to(device), pcl.to(device)
            outputs = model(rgb, depth, pcl)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    print(f"Test Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"New best model saved at {CHECKPOINT_PATH}")

print("Training complete.")