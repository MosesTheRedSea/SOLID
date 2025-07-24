



class PointCloudOnlyDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        _, _, pcl, label = self.base[idx]
        return pcl, label
