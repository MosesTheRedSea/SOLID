import os
import sys
sys.path.append("/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/scripts/")
import shutil
import torch
from tqdm import tqdm
from pymatreader import read_mat
import numpy as np43
from PIL import Image
import open3d as o3d
from constants import *
import os
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SUNRGBDDataset(Dataset):
    def __init__(self, data_root, toolbox_root, split='train', num_points=1024, transform_rgb=None, transform_depth=None,transform_points=None):
        self.data_root        = data_root
        self.num_points       = num_points
        self.transform_rgb    = transform_rgb
        self.transform_depth  = transform_depth
        self.transform_points = transform_points

        mat_path = os.path.join(toolbox_root, 'Metadata', 'SUNRGBDMeta.mat')
        print(f"Loading metadata from {mat_path}...")
        mat        = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        self.meta  = mat['SUNRGBDMeta']
        print(f"  → {len(self.meta)} samples in MAT")

        valid = []
        for i, sample in enumerate(self.meta):
            rgb_local   = self._to_local(sample.rgbpath)
            depth_local = self._to_local(sample.depthpath)
            if os.path.exists(rgb_local) and os.path.exists(depth_local):
                valid.append(i)
        print(f"  → {len(valid)} valid samples found")

        # 3) split
        cut = int(0.8 * len(valid))
        if split == 'train':
            self.sample_indices = valid[:cut]
        elif split == 'test':
            self.sample_indices = valid[cut:]
        else:
            raise ValueError("split must be 'train' or 'test'")
        print(f"Using {len(self.sample_indices)} samples for {split}")

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

        rgb_path  = self._to_local(sample.rgbpath)
        rgb_image = Image.open(rgb_path).convert('RGB')

        depth_path = self._to_local(sample.depthpath)
        depth_np   = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0

        if self.transform_depth:
            depth_tensor = self.transform_depth(depth_np)
        else:
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        K = getattr(sample, 'intrinsicMatrix', None) or getattr(sample, 'K', None)
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        pts, cols = self._depth_to_point_cloud(depth_np, rgb_image, fx, fy, cx, cy)

        # sample or pad to num_points
        if pts.shape[0] > self.num_points:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=False)
            pts, cols = pts[choice], cols[choice]
        else:
            pad = self.num_points - pts.shape[0]
            pts  = np.vstack([pts,  np.zeros((pad,3), dtype=np.float32)])
            cols = np.vstack([cols, np.zeros((pad,3), dtype=np.float32)])
        pcl = np.hstack([pts, cols]).astype(np.float32)

        # apply RGB & point‐cloud transforms
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)
        if self.transform_points:
            pcl = self.transform_points(pcl)

        return rgb_image, depth_tensor, torch.from_numpy(pcl)

    def _depth_to_point_cloud(self, depth, rgb, fx, fy, cx, cy):
        H, W   = depth.shape
        rgb_np = np.array(rgb) / 255.0
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        zs     = depth.flatten()
        mask   = (zs > 0) & (zs < 8.0)
        u      = us.flatten()[mask]
        v      = vs.flatten()[mask]
        z      = zs[mask]
        x      = (u - cx) * z / fx
        y      = (v - cy) * z / fy
        pts    = np.stack([x, y, z], axis=1)
        cols   = rgb_np[v, u]
        return pts, cols

if __name__ == "__main__":
    SUNRGBD_DATA_ROOT    = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBD"
    SUNRGBD_TOOLBOX_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data/sunrgbd/SUNRGBDtoolbox"

    for p in (SUNRGBD_DATA_ROOT, SUNRGBD_TOOLBOX_ROOT):
        if not os.path.exists(p):
            print(f"Error: '{p}' not found."); sys.exit(1)

    transform_rgb = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
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
    test_ds = SUNRGBDDataset(
        data_root=SUNRGBD_DATA_ROOT,
        toolbox_root=SUNRGBD_TOOLBOX_ROOT,
        split='test',
        num_points=1024,
        transform_rgb=transform_rgb,
        transform_depth=transform_depth
    )

    print(f"\nTraining set size: {len(train_ds)}")
    print(f"Test set size:     {len(test_ds)}\n")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    for i, (rgb, depth, pcl) in enumerate(loader):
        print(f"Batch {i+1}: RGB {rgb.shape}, Depth {depth.shape}, PCL {pcl.shape}")
        break
    print("Example batch loaded successfully.")