import os
import sys

sys.path.extend([
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/scripts/",    
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/",
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/models/", 
    "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/configs/baseline/"])

import shutil
import torch
from tqdm import tqdm
from pymatreader import read_mat
import numpy as np
from PIL import Image
import open3d as o3d
from constants import *
import os
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter

class SUNRGBDDataset(Dataset):
    def __init__(self, data_root, toolbox_root, split='train', num_points=1024,
             transform_rgb=None, transform_depth=None, transform_points=None,
             allowed_classes=None):
        self.data_root = data_root
        self.num_points = num_points
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.transform_points = transform_points
        self.allowed_classes = allowed_classes

        print(f"[{split.upper()}] Loading metadata...")
        mat = sio.loadmat(SUNRGBD_3DBB_MAT, squeeze_me=True, struct_as_record=False)
        self.meta = mat['SUNRGBDMeta']
        print(f"→ {len(self.meta)} total samples in dataset")

        # 1. Validate all samples & extract classes
        MINSAMPLES = 20
        valid = []
        label_counts = Counter()

        for i, sample in enumerate(self.meta):
            rgb_local = self._to_local(sample.rgbpath)
            depth_local = self._to_local(sample.depthpath)
            if not (os.path.exists(rgb_local) and os.path.exists(depth_local)):
                continue
            bbs = sample.groundtruth3DBB
            if bbs is None or (isinstance(bbs, np.ndarray) and bbs.size == 0):
                continue
            all_bbs = bbs if isinstance(bbs, np.ndarray) else [bbs]
            first = all_bbs[0]
            classname = first.classname
            if isinstance(classname, np.ndarray):
                classname = classname[0]
            label_counts[classname] += 1
            valid.append((i, classname))

        # 2. Keep only samples where class has enough instances
        filtered = [(i, cls) for i, cls in valid if label_counts[cls] >= MINSAMPLES]
        if self.allowed_classes is not None:
            filtered = [(i, cls) for i, cls in filtered if cls in self.allowed_classes]

        print(f"→ {len(filtered)} samples remaining after class frequency filtering (min {MINSAMPLES})")

        # 3. Sort classes and build mapping
        names = [cls for _, cls in filtered]
        self.classes = sorted(set(names))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 4. Split after filtering
        np.random.seed(42)
        np.random.shuffle(filtered)
        cut = int(0.8 * len(filtered))
        if split == 'train':
            selected = filtered[:cut]
        elif split == 'test':
            selected = filtered[cut:]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.sample_indices = [i for i, _ in selected]
        print(f"→ Using {len(self.sample_indices)} samples for {split}")
        print(f"→ {len(self.classes)} classes after filtering (min {MINSAMPLES} samples)")

    def _to_local(self, full_path):
        p = full_path.replace('\\', '/')
        if p.startswith(ORIG_PREFIX):
            p = p[len(ORIG_PREFIX):]
        p = p.lstrip('/\\')
        return os.path.join(self.data_root, p)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample = self.meta[self.sample_indices[idx]]

        rgb_path = self._to_local(sample.rgbpath)
        rgb_image = Image.open(rgb_path).convert('RGB')

        depth_path = self._to_local(sample.depthpath)
        depth_np = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        depth_tensor = self.transform_depth(depth_np) if self.transform_depth else torch.from_numpy(depth_np).unsqueeze(0)

        K = getattr(sample, 'intrinsicMatrix', None) or getattr(sample, 'K', None)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts, cols = self._depth_to_point_cloud(depth_np, rgb_image, fx, fy, cx, cy)
        if pts.shape[0] > self.num_points:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=False)
            pts, cols = pts[choice], cols[choice]
        else:
            pad = self.num_points - pts.shape[0]
            pts = np.vstack([pts, np.zeros((pad, 3), dtype=np.float32)])
            cols = np.vstack([cols, np.zeros((pad, 3), dtype=np.float32)])
        pcl = torch.from_numpy(pts.astype(np.float32))

        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)
        if self.transform_points:
            pcl = self.transform_points(pcl)

        bbs = sample.groundtruth3DBB
        all_bbs = bbs if isinstance(bbs, np.ndarray) else [bbs]
        raw = all_bbs[0].classname
        if isinstance(raw, np.ndarray):
            raw = raw[0]
        label = torch.tensor(self.class_to_idx[raw], dtype=torch.long)
        scene_type = getattr(sample, 'sceneType', 'unknown')
        return rgb_image, depth_tensor, pcl, label, scene_type

    def _depth_to_point_cloud(self, depth, rgb, fx, fy, cx, cy):
        
        H, W = depth.shape

        rgb_np = np.array(rgb) / 255.0
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        zs = depth.flatten()

        mask = (zs > 0) & (zs < 8.0)
        u = us.flatten()[mask]
        v = vs.flatten()[mask]
        z = zs[mask]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts = np.stack([x, y, z], axis=1)
        cols = rgb_np[v, u]

        return pts, cols

if __name__ == "__main__":
    SUNRGBD_DATA_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBD"
    SUNRGBD_TOOLBOX_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBDtoolbox"

    for p in (SUNRGBD_DATA_ROOT, SUNRGBD_TOOLBOX_ROOT):
        if not os.path.exists(p):
            print(f"Error: '{p}' not found."); sys.exit(1)

    transform_rgb = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std =[0.229,0.224,0.225])])

    transform_depth = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_ds = SUNRGBDDataset(
        data_root=SUNRGBD_DATA_ROOT,
        toolbox_root=SUNRGBD_TOOLBOX_ROOT,
        split='train',
        num_points=1024,
        transform_rgb=transform_rgb,
        transform_depth=transform_depth
    )

    print("\n")

    test_ds = SUNRGBDDataset(
        data_root=SUNRGBD_DATA_ROOT,
        toolbox_root=SUNRGBD_TOOLBOX_ROOT,
        split='test',
        num_points=1024,
        transform_rgb=transform_rgb,
        transform_depth=transform_depth,
        allowed_classes=train_ds.classes
    )

    assert train_ds.class_to_idx == test_ds.class_to_idx, \
    "Label index mapping between train and test is different!"


    print(f"\nTraining set size: {len(train_ds)}")
    print(f"Test set size: {len(test_ds)}\n")

    print("[SUMMARY]")
    print(f"Train classes ({len(train_ds.classes)}): {train_ds.classes}\n")
    print(f"Test classes  ({len(test_ds.classes)}):  {test_ds.classes}")

    # Check if test classes are all in train
    missing = set(test_ds.classes) - set(train_ds.classes)

    if missing:
        print(f"\n Test set contains {len(missing)} unseen classes: {missing}")
    else:
        print("\n All test classes are present in the training set.")
    
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    for i, (rgb, depth, pcl, label, scene_type) in enumerate(loader):
        print(f"Batch {i+1}: RGB {rgb.shape}, Depth {depth.shape}, PCL {pcl.shape}, Labels {label.shape}")
        break

    print("Example batch loaded successfully.")
