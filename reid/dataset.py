# reid/dataset.py
import os, random
from PIL import Image
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def is_image_file(fname):
    _ext = os.path.splitext(fname)[1].lower()
    return _ext in IMG_EXT

class ReIDDataset(Dataset):
    """
    dataset expects a list of tuples (img_path, pid).
    Returns (img_tensor, pid)
    """
    def __init__(self, samples: List[Tuple[str,int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, pid

def make_dataset_from_folder(root_dir: str):
    """
    root_dir/
        fish1/
            img...
        fish2/
    returns list of (path, pid) and a dict mapping classname->pid
    """
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    samples = []
    cls2id = {}
    for i,cls in enumerate(classes):
        cls2id[cls] = i
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if is_image_file(fname):
                samples.append((os.path.join(cls_dir, fname), i))
    return samples, cls2id
