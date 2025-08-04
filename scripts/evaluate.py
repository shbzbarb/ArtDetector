"""
Script for evaluating a trained checkpoint on the test set

Reports:
- Top-1 accuracy
- Macro-F1
- Expected Calibration Error (ECE) using the saved temperature
- Saves a confusion matrix image to logs/plots/confusion_matrix.png

Run (from project root):
    python -m scripts.evaluate
Optionally:
    python -m scripts.evaluate --ckpt checkpoints/best_art_style_classifier.pth
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import cfg
from utils.dataset_utils import make_datasets
from utils.metrics import expected_calibration_error, plot_confusion_matrix
from models.resnet50_transfer import ResNet50TL

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=cfg.CHECKPOINTS / "best_art_style_classifier.pth")
    parser.add_argument("--cm_out", type=Path, default=cfg.PLOTS_DIR / "confusion_matrix.png")
    parser.add_argument("--roc_out", type=Path, default=cfg.PLOTS_DIR / "roc_curve.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Load checkpoint
    assert args.ckpt.exists(), f"Checkpoint not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location=device)
    num_classes = ckpt["num_classes"]
    img_size, mean, std = ckpt["img_size"], ckpt["mean"], ckpt["std"]
    temperature = float(ckpt.get("temperature", 1.0))

    #Datasets/loaders (only test is required)
    _, _, test_ds = make_datasets(cfg.TRAIN_DIR, cfg.VAL_DIR, cfg.TEST_DIR, img_size, mean, std)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)

    #Model
    model = ResNet50TL(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_logits, all_labels = [], []
    pbar = tqdm(test_loader, desc="Eval (test)")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(imgs) / temperature
        all_logits.append(logits)
        all_labels.append(labels)

    logits = torch.cat(all_logits) if all_logits else torch.empty(0, num_classes, device=device)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long, device=device)

    #Acc / F1 / ECE
    preds = logits.argmax(1) if logits.numel() else torch.empty(0, dtype=torch.long, device=device)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0
    f1 = f1_score(y_true, y_pred, average="macro") if y_true.size else 0.0
    ece = expected_calibration_error(logits, labels, n_bins=15) if y_true.size else 0.0
    print(f"Test Acc: {acc:.4f} | Macro-F1: {f1:.4f} | ECE: {ece:.4f}")

    #Confusion matrix
    args.cm_out.parent.mkdir(parents=True, exist_ok=True)
    if y_true.size:
        plot_confusion_matrix(y_true, y_pred, test_ds.classes, save_path=args.cm_out)
        print(f"Saved confusion matrix to {args.cm_out}")

    #ROC AUC (multi-class)
    try:
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fpr, tpr, roc_auc = {}, {}, {}
        valid_classes = []
        for i in range(num_classes):
            pos = y_bin[:, i].sum()
            if pos == 0 or pos == y_bin.shape[0]:
                continue 
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            valid_classes.append(i)

        macro_auc = float(np.mean([roc_auc[i] for i in valid_classes])) if valid_classes else float("nan")
        #micro ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
        micro_auc = auc(fpr_micro, tpr_micro)

        print(f"ROC AUC — micro: {micro_auc:.4f} | macro (valid classes): {macro_auc:.4f}")

        #plot ROC curves
        fig, ax = plt.subplots(figsize=(9, 7))
        #micro
        ax.plot(fpr_micro, tpr_micro, linewidth=2, label=f"micro-average (AUC={micro_auc:.3f})")
        #per-class
        for i in valid_classes:
            ax.plot(fpr[i], tpr[i], linewidth=1, alpha=0.8, label=f"{test_ds.classes[i]} (AUC={roc_auc[i]:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        fig.tight_layout()
        args.roc_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.roc_out, dpi=150)
        plt.close(fig)
        print(f"Saved ROC curve to {args.roc_out}")
    except Exception as e:
        print(f"ROC computation skipped: {e}")

if __name__ == "__main__":
    main()