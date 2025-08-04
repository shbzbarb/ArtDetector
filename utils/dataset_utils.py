from pathlib import Path
from typing import Tuple, List
from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def _maybe_randaugment():
    try:
        return transforms.RandAugment(num_ops=2, magnitude=8)
    except AttributeError:
        class Identity:
            def __call__(self, x): return x
        return Identity()

def build_transforms(img_size: int, mean, std):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        _maybe_randaugment(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean, std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, eval_tfms

def pil_loader_safe(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def _filter_corrupt_samples(ds: datasets.ImageFolder, desc: str):
    good, bad = [], 0
    for path, target in tqdm(ds.samples, desc=f"Verifying {desc}", leave=False):
        try:
            with Image.open(path) as img:
                img.verify()
            good.append((path, target))
        except Exception:
            bad += 1
    if bad:
        tqdm.write(f"[{desc}] removed {bad} corrupt/unreadable files")
    ds.samples = good
    if hasattr(ds, "imgs"): ds.imgs = good
    ds.targets = [t for _, t in good]
    return bad

def make_datasets(train_dir: Path, val_dir: Path, test_dir: Path,
                  img_size: int, mean, std):
    train_tfms, eval_tfms = build_transforms(img_size, mean, std)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms, loader=pil_loader_safe)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tfms,  loader=pil_loader_safe)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tfms,  loader=pil_loader_safe)
    _filter_corrupt_samples(train_ds, "train")
    _filter_corrupt_samples(val_ds,   "val")
    _filter_corrupt_samples(test_ds,  "test")
    return train_ds, val_ds, test_ds

def _make_weighted_sampler(targets, num_classes: int):
    counts = Counter(targets)
    counts = {c: max(1, counts.get(c, 0)) for c in range(num_classes)}
    weights_by_class = {c: 1.0 / counts[c] for c in range(num_classes)}
    sample_weights = torch.DoubleTensor([weights_by_class[t] for t in targets])
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def make_loaders(train_ds, val_ds, test_ds,
                 batch_size: int, num_workers: int,
                 use_weighted_sampler: bool = True):
    targets = getattr(train_ds, "targets", [t for _, t in getattr(train_ds, "samples")])
    if use_weighted_sampler:
        sampler = _make_weighted_sampler(targets, num_classes=len(train_ds.classes))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader