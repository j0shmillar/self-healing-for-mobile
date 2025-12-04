"""
Trains a ResNet-18 head on Mini-ImageNet using MLclf.miniimagenet_clf_dataset, from ImageNet-pretrained weights.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from MLclf import MLclf

from tqdm import tqdm


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes_from_dataset(dset):
    """
    Infer number of classes from dataset labels.
    Assumes __getitem__ returns (img, label) with label as int or tensor.
    """
    labels = set()
    for idx in range(len(dset)):
        _, y = dset[idx]
        y = int(y) if not isinstance(y, int) else y
        labels.add(y)
    num_classes = max(labels) + 1
    return num_classes


# ---------------------------
# Main training function
# ---------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_seed(args.seed)

    print("Loading Mini-ImageNet via MLclf...")
    base_tf = transforms.Compose([
        transforms.ToTensor(),   # images as [C,H,W] in [0,1]
    ])

    train_dset, val_dset, _ = MLclf.miniimagenet_clf_dataset(
        ratio_train=args.ratio_train,
        ratio_val=args.ratio_val,
        seed_value=args.seed,
        shuffle=True,
        transform=base_tf,
        save_clf_data=True,
    )

    print(f"Train size: {len(train_dset)}, Val size: {len(val_dset)}")

    if args.num_classes is None:
        num_classes = get_num_classes_from_dataset(train_dset)
        print(f"Inferred num_classes = {num_classes}")
    else:
        num_classes = args.num_classes
        print(f"Using provided num_classes = {num_classes}")

    # Dataloaders
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model: ResNet-18 pretrained on ImageNet
    print("Building ResNet-18 (ImageNet-pretrained) ...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all backbone params
    for p in resnet.parameters():
        p.requires_grad = False

    # Replace head
    resnet.fc = nn.Linear(512, num_classes)
    resnet.fc.weight.data.normal_(0, 0.01)
    resnet.fc.bias.data.zero_()

    resnet.to(device)

    # Only optimize head
    optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Preprocess to match your runtime Model
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def run_epoch(loader, train_mode=True):
        if train_mode:
            resnet.train()
        else:
            resnet.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            imgs = preprocess(imgs)

            if train_mode:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train_mode):
                logits = resnet(imgs)
                loss = criterion(logits, labels)

                if train_mode:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    best_val_acc = 0.0
    best_state = None

    print("Starting training...")
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss, train_acc = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc = run_epoch(val_loader, train_mode=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = resnet.state_dict()
            torch.save(best_state, args.out_path)
            print(f"  -> New best val_acc={best_val_acc:.4f}, saved to {args.out_path}")

    print(f"Done. Best val_acc={best_val_acc:.4f} (checkpoint: {args.out_path})")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ratio_train", type=float, default=0.6)
    parser.add_argument("--ratio_val", type=float, default=0.2)

    parser.add_argument("--num_classes", type=int, default=None,
                        help="If None, infer from dataset labels; else use this value")

    parser.add_argument("--out_path", type=str, default="miniimagenet_resnet18.pth")

    args = parser.parse_args()
    train(args)

