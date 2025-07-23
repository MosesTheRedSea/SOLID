#!/usr/bin/env python
import os
import sys
from collections import Counter

module_directory = os.path.abspath('/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/data')

SCRIPT_DIR   = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID/scripts"
PROJECT_ROOT = "/home/hice1/madewolu9/scratch/madewolu9/SOLID/SOLID"



def main():
    data_root = os.path.join(PROJECT_ROOT, "data", "sunrgbd")
    ds = SUNRGBDRGB(data_root, transform=None)

    total = len(ds)
    classes = ds.classes
    counts = Counter(label for _, label in ds.samples)

    print(f"Project root: {PROJECT_ROOT}")

    print(f"Data root:    {data_root}")

    print(f"Total images: {total}")

    print(f"Num classes:  {len(classes)}")

    print("\nSamples per class:")

    for idx, cls in enumerate(classes):
        print(f"  {cls:15s}: {counts[idx]}")

    # I split the datasets 80 | 20
    val_sz = int(0.2 * total)

    train_sz = total - val_sz

    print(f"\n80/20 split → train: {train_sz}, val: {val_sz}")

if __name__ == "__main__":
    main()
