# reid/train_reid.py
import os, sys, argparse, time
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from reid.dataset import make_dataset_from_folder, ReIDDataset
from reid.sampler import PKSampler
from reid.models import ReIDModel
from reid.losses import BatchHardTripletLoss
# optimizer
try:
    from torch.optim import AdamW as OptimAdamW
except Exception:
    from torch.optim import Adam as OptimAdamW  # fallback to Adam if AdamW not available
from torch.optim import Adam, SGD

# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

EPS = 1e-9

def prepare_split(data_root, out_dir, val_ratio=0.2, seed=42):
    """
    create train_list.txt and val_list.txt under out_dir
    """
    samples, cls2id = make_dataset_from_folder(data_root)
    random.seed(seed)
    by_pid = {}
    for p, pid in samples:
        by_pid.setdefault(pid, []).append(p)
    train_lines = []
    val_lines = []
    for pid, imgs in by_pid.items():
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]
        for p in train_imgs:
            train_lines.append(f"{p} {pid}")
        for p in val_imgs:
            val_lines.append(f"{p} {pid}")
    os.makedirs(out_dir, exist_ok=True)
    train_list = os.path.join(out_dir, "train_list.txt")
    val_list = os.path.join(out_dir, "val_list.txt")
    with open(train_list, "w") as f:
        f.write("\n".join(train_lines))
    with open(val_list, "w") as f:
        f.write("\n".join(val_lines))
    print("Wrote", train_list, val_list)
    return train_list, val_list

def load_list(list_path):
    samples = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p, pid = line.split()
            samples.append((p, int(pid)))
    return samples

def build_transforms():
    train_t = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1,0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.3), ratio=(0.3,3.3), value=0)  # helps ReID generalization
    ])
    val_t = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_t, val_t

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.prepare_split:
        prepare_split(args.data_root, args.out_dir, val_ratio=args.val_ratio, seed=args.seed)

    train_list = os.path.join(args.out_dir, "train_list.txt")
    val_list = os.path.join(args.out_dir, "val_list.txt")
    if not os.path.exists(train_list) or not os.path.exists(val_list):
        raise FileNotFoundError("train_list.txt or val_list.txt not found in out_dir. Run with --prepare_split first or create them.")

    train_samples = load_list(train_list)
    val_samples = load_list(val_list)
    num_classes = len(set([pid for _,pid in train_samples]))
    print(f"Found {len(train_samples)} train samples, {len(val_samples)} val samples, classes: {num_classes}")

    train_t, val_t = build_transforms()
    train_dataset = ReIDDataset(train_samples, transform=train_t)
    val_dataset = ReIDDataset(val_samples, transform=val_t)

    # PKSampler params: P * K = batch size
    P = min(args.P, len(set([pid for _,pid in train_samples])))
    K = args.K
    batch_size = P * K
    sampler = PKSampler(train_samples, batch_size=batch_size, P=P, K=K)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, 64), shuffle=False, num_workers=args.num_workers)

    # model
    model = ReIDModel(num_classes=num_classes, emb_dim=args.emb_dim, pretrained_backbone=None, use_imagenet_pretrained=True)
    model = model.to(device)

    # optionally load user-provided backbone weights (not used in Option A by default)
    if args.pretrained_backbone is not None:
        try:
            model._load_backbone_weights(args.pretrained_backbone)
        except Exception as e:
            print("Warning: failed to load custom backbone weights:", e)

    # freeze backbone initially if requested
    freeze_epochs = args.freeze_backbone_epochs
    if freeze_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"Backbone frozen for first {freeze_epochs} epochs.")

    # create optimizer for trainable params (initial stage)
    def create_optimizer(model, head_lr, backbone_lr=None, use_adamw=True):
        head_params = []
        backbone_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # treat fc, bn, classifier as head
            if "fc" in name or "bn" in name or "classifier" in name:
                head_params.append(p)
            else:
                backbone_params.append(p)
        params = []
        if head_params:
            params.append({'params': head_params, 'lr': head_lr})
        if backbone_params and backbone_lr is not None:
            params.append({'params': backbone_params, 'lr': backbone_lr})
        if use_adamw:
            try:
                opt = OptimAdamW(params, lr=head_lr, weight_decay=args.weight_decay)
            except Exception:
                opt = Adam(params, lr=head_lr, weight_decay=args.weight_decay)
        else:
            opt = SGD(params, lr=head_lr, momentum=0.9, weight_decay=args.weight_decay)
        return opt

    # initial optimizer: only head params if frozen; else both with different lrs
    if freeze_epochs > 0:
        optimizer = create_optimizer(model, head_lr=args.head_lr, backbone_lr=None, use_adamw=args.use_adamw)
    else:
        optimizer = create_optimizer(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr, use_adamw=args.use_adamw)

    # scheduler: will be re-created after unfreeze to anneal over remaining epochs
    remaining_epochs_after_unfreeze = max(1, args.epochs - freeze_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))  # initial scheduler, will be reset on unfreeze

    # loss
    ce_loss = nn.CrossEntropyLoss().to(device)
    tri_loss = BatchHardTripletLoss(margin=args.margin).to(device)

    best_val = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        # handle unfreeze point
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print("Unfreezing backbone for fine-tuning (recreating optimizer & scheduler).")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = create_optimizer(model, head_lr=args.head_lr, backbone_lr=args.backbone_lr, use_adamw=args.use_adamw)
            # scheduler: anneal over remaining epochs
            remaining = max(1, args.epochs - epoch + 1)
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

        t0 = time.time()
        running_loss = 0.0
        running_ce = 0.0
        running_tri = 0.0
        iters = 0
        for i, batch in enumerate(train_loader):
            imgs, pids = batch
            imgs = imgs.to(device)
            pids = pids.to(device)
            logits, feats = model(imgs)
            loss_ce = ce_loss(logits, pids)
            loss_tri = tri_loss(feats, pids)
            loss = loss_ce + args.lambda_tri * loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_tri += loss_tri.item()
            iters += 1
        # step scheduler at epoch end
        try:
            scheduler.step()
        except Exception:
            pass
        t1 = time.time()

        print(f"Epoch {epoch}/{args.epochs} train_loss={running_loss/iters:.4f} ce={running_ce/iters:.4f} tri={running_tri/iters:.4f} time={t1-t0:.1f}s lr_head={args.head_lr} lr_backbone={args.backbone_lr}")

        # val (classification acc)
        model.eval()
        total = 0; correct = 0
        with torch.no_grad():
            for imgs, pids in val_loader:
                imgs = imgs.to(device); pids = pids.to(device)
                logits, feats = model(imgs)
                preds = torch.argmax(logits, dim=1)
                total += preds.size(0)
                correct += (preds == pids).sum().item()
        acc = correct / total if total>0 else 0.0
        print(f"Val classification acc: {acc:.4f}")

        # save checkpoint every epoch
        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"reid_epoch{epoch}.pth")
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)

        if acc > best_val:
            best_val = acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, "reid_best.pth"))
            print("Saved best model (classification acc).")

    print("Training finished. Best val acc:", best_val, "at epoch", best_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="root folder with fishX subfolders")
    parser.add_argument("--out_dir", required=True, help="outputs: train/val lists, ckpts")
    parser.add_argument("--prepare_split", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_backbone", default=None, help="optional path to backbone weights (not required for Option A)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--head_lr", type=float, default=1e-3, help="learning rate for head (fc/bn/classifier)")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="learning rate for backbone when unfreezing")
    parser.add_argument("--P", type=int, default=9)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--lambda_tri", type=float, default=1.0)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5, help="freeze backbone for N epochs first")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_adamw", action="store_true", help="use AdamW optimizer if available (recommended)")
    args = parser.parse_args()
    train(args)
