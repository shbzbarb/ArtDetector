import argparse, os, sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import cfg
from models.resnet50_transfer import ResNet50TL

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def collect_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        return [path]
    if path.is_dir():
        return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    raise FileNotFoundError(f"No file or directory found at {path}")

def build_eval_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def tta_batches(img: Image.Image, img_size: int, mean, std, n: int = 6) -> List[torch.Tensor]:
    """
    Return a list of TENSORS, one per scale. Each tensor is a batch [N_scale, C, H, W]
    with consistent H×W within that tensor. Caller averages logits across scales.
    """
    sizes = [img_size, int(img_size * 1.14)]
    out_batches: List[torch.Tensor] = []
    per_scale = max(1, n // len(sizes))
    for s in sizes:
        tfm = build_eval_transform(s, mean, std)
        xs = []
        for i in range(per_scale):
            x = tfm(img)
            if i % 2 == 1:
                x = torch.flip(x, dims=[2])
            xs.append(x)
        out_batches.append(torch.stack(xs, dim=0))
    return out_batches

def load_checkpoint(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet50TL(num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    meta = {
        "img_size": ckpt["img_size"],
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "temperature": float(ckpt.get("temperature", 1.0)),
        "idx_to_class": idx_to_class,
    }
    return model, meta

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Image file or directory")
    parser.add_argument("--ckpt", type=Path, default=cfg.CHECKPOINTS / "best_art_style_classifier.pth")
    parser.add_argument("--conf", type=float, default=cfg.CONF_THRESH)
    parser.add_argument("--tta", type=int, default=cfg.TTA_N)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.ckpt, device)
    img_size, mean, std = meta["img_size"], meta["mean"], meta["std"]
    T = meta["temperature"]
    idx_to_class = meta["idx_to_class"]

    paths = collect_images(args.path)
    print(f"Found {len(paths)} image(s). Using device: {device} | TTA={args.tta} | T={T:.3f}")

    for img_path in tqdm(paths, desc="Predicting"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to open {img_path}: {e}")
            continue

        batches = tta_batches(img, img_size, mean, std, n=max(2, args.tta))
        logits_sum = None
        n_parts = 0
        for xb in batches:
            xb = xb.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits_part = model(xb).mean(0, keepdim=True)
            logits_sum = logits_part if logits_sum is None else (logits_sum + logits_part)
            n_parts += 1

        logits = (logits_sum / max(1, n_parts)) / T
        probs = F.softmax(logits, dim=1).squeeze(0)

        conf, pred_idx = torch.max(probs, dim=0)
        conf_val = conf.item()
        pred_class = idx_to_class[pred_idx.item()]

        topk = torch.topk(probs, k=min(args.topk, probs.numel()))
        topk_pairs = [(idx_to_class[i.item()], probs[i].item()) for i in topk.indices]

        print(f"\n{img_path}")
        if conf_val >= args.conf:
            print(f"  PREDICTION: {pred_class} | confidence={conf_val:.3f}")
        else:
            print(f"  LOW CONFIDENCE (<{args.conf:.2f}). Top-{len(topk_pairs)}:")
            for c, p in topk_pairs:
                print(f"   - {c}: {p:.3f}")

if __name__ == "__main__":
    main()
