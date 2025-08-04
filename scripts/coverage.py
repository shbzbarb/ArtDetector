"""
Computes coverage and accuracy among high-confidence predictions

Usage:
python -m scripts.coverage --ckpt checkpoints/best_art_style_classifier.pth --conf 0.80
"""

import argparse, os, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

# project imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import cfg
from utils.dataset_utils import make_datasets
from models.resnet50_transfer import ResNet50TL

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=cfg.CHECKPOINTS / "best_art_style_classifier.pth")
    ap.add_argument("--conf", type=float, default=cfg.CONF_THRESH,
                    help="confidence threshold")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    num_classes = ckpt["num_classes"]
    img_size, mean, std = ckpt["img_size"], ckpt["mean"], ckpt["std"]
    T = float(ckpt.get("temperature", 1.0))

    #loaders (only test)
    _, _, test_ds = make_datasets(cfg.TRAIN_DIR, cfg.VAL_DIR, cfg.TEST_DIR,
                                  img_size, mean, std)
    loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)

    #model
    model = ResNet50TL(num_classes=num_classes, pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    covered = correct = total = 0
    for imgs, labels in tqdm(loader, desc="Coverage"):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            probs = F.softmax(model(imgs) / T, dim=1)
        conf, pred = probs.max(1)
        mask = conf >= args.conf
        covered += mask.sum().item()
        correct += (pred[mask] == labels[mask]).sum().item()
        total += labels.size(0)

    cov = covered / total if total else 0.0
    acc_in = correct / covered if covered else 0.0
    print(f"Threshold  : {args.conf:.2f}")
    print(f"Coverage   : {cov:.3%}  ({covered}/{total})")
    print(f"Acc (covered set) : {acc_in:.3%}")

if __name__ == "__main__":
    main()