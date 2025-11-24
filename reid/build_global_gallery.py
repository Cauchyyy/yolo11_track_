#!/usr/bin/env python3
"""
reid/build_global_gallery.py

给定 data_root 下的子文件夹（每个子文件夹为一个 ID/class，例如 fish1, fish2 ...），
对每张图片用 backbone(resnet50) 提取 embedding（默认为 2048-d），
对每个 class 做 L2-normalized 平均，得到 global gallery (num_classes x D).
同时输出一些诊断统计信息。

示例：
python reid/build_global_gallery.py \
  --data_root /home/waas/ultralytics-yolo11-main/fish_dataset \
  --out_npy /home/waas/reid_out/global_gallery.npy \
  --out_meta /home/waas/reid_out/global_gallery_meta.json \
  --backbone /home/waas/ultralytics-yolo11-main/reid_out/backbone_resnet50.pth \
  --batch_size 32 --image_size 224 --device cuda:0 --save_embeddings_dir /home/waas/reid_out/embs

Requirements:
  torch, torchvision, numpy, pillow, tqdm
"""
import os, sys, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# ---------------------------
# Dataset
# ---------------------------
class FolderDataset(Dataset):
    def __init__(self, root_dir, exts=('.jpg','.jpeg','.png','.bmp'), transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            p = self.root_dir / c
            for img_path in sorted(p.iterdir()):
                if img_path.suffix.lower() in exts:
                    self.samples.append((str(img_path), self.class_to_idx[c], c))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir}. Expected structure: {root_dir}/class_x/*.jpg")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p, lbl, clsname = self.samples[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, lbl, p, clsname

# ---------------------------
# Model (backbone)
# ---------------------------
def build_backbone(resnet_variant='resnet50', pretrained=False, out_dim=2048):
    if resnet_variant == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError("Only resnet50 supported in this script")
    # remove the final fc, keep global avg pool output (2048)
    modules = list(model.children())[:-1]  # remove fc
    trunk = nn.Sequential(*modules)
    # trunk outputs (B, 2048, 1, 1) => we'll squeeze to (B,2048)
    return trunk

def load_backbone_weights(trunk, weight_path, strict=False, map_location='cpu'):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weight file not found: {weight_path}")
    sd = torch.load(weight_path, map_location=map_location)
    # handle possible nested dict like {'model': state_dict}
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    # heuristics: strip 'module.' or 'backbone.' prefixes, and keep only keys that match
    new_sd = {}
    for k,v in sd.items():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        # if weight was saved for full model with 'backbone.' prefix
        if new_k.startswith('backbone.'):
            new_k = new_k[len('backbone.'):]
        new_sd[new_k] = v
    # try to load matching keys only
    model_sd = trunk.state_dict()
    compatible = {k:new_sd[k] for k in new_sd.keys() if k in model_sd and new_sd[k].shape == model_sd[k].shape}
    if len(compatible) == 0:
        print("Warning: found 0 compatible keys between provided weights and trunk. Loading skipped.")
    else:
        model_sd.update(compatible)
        trunk.load_state_dict(model_sd)
        print(f"Loaded {len(compatible)}/{len(model_sd)} matching params into backbone (strict={strict})")
    return trunk

# ---------------------------
# Utility: forward batch, extract features
# ---------------------------
def extract_batch_feats(trunk, images_tensor, device='cpu'):
    # trunk returns (B, C, 1, 1)
    with torch.no_grad():
        x = images_tensor.to(device)
        out = trunk(x)  # (B, C, 1, 1)
        out = out.view(out.size(0), -1).cpu().numpy()
    return out

# ---------------------------
# Main build function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="root folder with class subfolders (fish1..fish9)")
    parser.add_argument("--out_npy", required=True, help="output npy path for gallery prototypes (num_classes x D)")
    parser.add_argument("--out_meta", required=False, default=None, help="json meta output path (class order, counts, dims)")
    parser.add_argument("--backbone", default=None, help="path to backbone weights (optional)")
    parser.add_argument("--device", default="cuda:0", help="device")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_mean", nargs=3, type=float, default=(0.485,0.456,0.406))
    parser.add_argument("--image_std", nargs=3, type=float, default=(0.229,0.224,0.225))
    parser.add_argument("--save_embeddings_dir", default=None, help="if given, save per-image embeddings under this folder (optional)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--norm_prototypes", action='store_true', help="L2-normalize prototypes before saving (recommended)")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # transforms
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=args.image_mean, std=args.image_std)
    ])

    dataset = FolderDataset(args.data_root, transform=transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model
    trunk = build_backbone('resnet50', pretrained=False)
    trunk.to(device).eval()
    if args.backbone:
        try:
            trunk = load_backbone_weights(trunk, args.backbone, strict=False, map_location=device)
            trunk.to(device).eval()
        except Exception as e:
            print("Warning: failed to load backbone weights:", e)

    # prepare accumulators
    embeddings_by_class = defaultdict(list)
    counts_by_class = defaultdict(int)
    image_paths = []
    image_embs = []

    print(f"Dataset samples: {len(dataset)}, classes: {len(class_to_idx)}")
    # iterate and extract
    for batch in tqdm(loader, desc="Extracting embeddings"):
        imgs, labels, paths, clsnames = batch
        # imgs: tensor (B,C,H,W)
        feats = extract_batch_feats(trunk, imgs, device=device)  # (B, D)
        for i in range(feats.shape[0]):
            lbl = int(labels[i].item())
            embeddings_by_class[lbl].append(feats[i])
            counts_by_class[lbl] += 1
            image_paths.append(paths[i])
            image_embs.append(feats[i])
    # compute per-class prototype (average + normalize)
    class_ids = sorted(list(class_to_idx.values()))
    prototypes = []
    meta = {'classes': [], 'counts': {}, 'dim': None}
    for cid in class_ids:
        arr = np.vstack(embeddings_by_class[cid]) if len(embeddings_by_class[cid])>0 else np.zeros((1, feats.shape[1]))
        proto = np.mean(arr, axis=0)
        if args.norm_prototypes:
            proto = proto / (np.linalg.norm(proto) + 1e-9)
        prototypes.append(proto.astype(np.float32))
        meta['classes'].append(idx_to_class[cid])
        meta['counts'][idx_to_class[cid]] = int(counts_by_class[cid])
        meta['dim'] = int(proto.shape[0])
    prototypes = np.vstack(prototypes)  # (num_classes, D)
    # ensure normalized (recommended)
    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)

    # save npy
    os.makedirs(os.path.dirname(args.out_npy) or '.', exist_ok=True)
    np.save(args.out_npy, prototypes)
    print("Saved gallery npy:", args.out_npy, "shape:", prototypes.shape)
    if args.out_meta:
        meta_out = dict(meta)
        meta_out['class_to_idx'] = class_to_idx
        with open(args.out_meta, 'w') as f:
            json.dump(meta_out, f, indent=2)
        print("Saved meta json:", args.out_meta)

    # optional: save per-image embeddings (same order as dataset.samples)
    if args.save_embeddings_dir:
        os.makedirs(args.save_embeddings_dir, exist_ok=True)
        embs_arr = np.vstack(image_embs) if len(image_embs)>0 else np.zeros((0, prototypes.shape[1]))
        np.save(os.path.join(args.save_embeddings_dir, 'image_embeddings.npy'), embs_arr)
        # save mapping of paths -> index
        with open(os.path.join(args.save_embeddings_dir, 'image_paths.txt'), 'w') as f:
            for p in image_paths:
                f.write(p + '\n')
        print("Saved per-image embeddings to:", args.save_embeddings_dir)

    # diagnostics: pairwise prototype cosine similarity matrix
    proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    sim = np.dot(proto_norm, proto_norm.T)
    print("Prototype cosine-similarity matrix shape:", sim.shape)
    if args.verbose:
        print(sim)
    # compute minimal inter-class similarity (off-diagonal max) and avg
    n = sim.shape[0]
    offdiag = sim[~np.eye(n, dtype=bool)].reshape(n, n-1)
    mean_off = float(np.mean(offdiag))
    max_off = float(np.max(offdiag))
    print(f"Prototype mean inter-class similarity: {mean_off:.4f}, max inter-class similarity: {max_off:.4f}")
    # nearest neighbor for each prototype
    nearest = np.argmax(sim - np.eye(n)*10.0, axis=1)
    for i in range(n):
        print(f"Proto {i} ({meta['classes'][i]}): nearest proto {nearest[i]} ({meta['classes'][nearest[i]]}), sim {sim[i, nearest[i]]:.4f}")

if __name__ == "__main__":
    main()
