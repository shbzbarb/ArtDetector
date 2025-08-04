"""
Transfer-learning trainer with automatic saving of loss / accuracy plots

After training finishes and the checkpoint is saved, the script automatically calls `python -m scripts.plot_training`, which reads
logs/training_logs.csv and writes: logs/plots/loss_curves.png, logs/plots/val_accuracy.png
"""

import argparse, os, sys, time, subprocess
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# project imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import cfg
from utils.logger import CSVLogger
from utils.dataset_utils import make_datasets, make_loaders
from utils.metrics import expected_calibration_error
from models.resnet50_transfer import (
    ResNet50TL,
    freeze_backbone,
    unfreeze_all,
    TemperatureScaler,
)

#utils
def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, clip=None):
    model.train(); running, n = 0.0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.MIXED_PRECISION):
            logits = model(imgs); loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer); scaler.update()
        running += loss.item() * imgs.size(0); n += imgs.size(0)
    return running / max(1, n)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval(); tot, n = 0.0, 0; all_logits, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.MIXED_PRECISION):
            logits = model(imgs); loss = criterion(logits, labels)
        tot += loss.item() * imgs.size(0); n += imgs.size(0)
        all_logits.append(logits); all_labels.append(labels)
    if n == 0:
        return 0.0, 0.0, 0.0, torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    logits = torch.cat(all_logits); labels = torch.cat(all_labels)
    acc = (logits.argmax(1) == labels).float().mean().item()
    ece = expected_calibration_error(logits, labels, n_bins=15)
    return tot / n, acc, ece, logits, labels

#main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=cfg.CHECKPOINTS / "best_art_style_classifier.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed); torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = make_datasets(
        cfg.TRAIN_DIR, cfg.VAL_DIR, cfg.TEST_DIR, cfg.IMG_SIZE, cfg.MEAN, cfg.STD
    )
    train_loader, val_loader, _ = make_loaders(
        train_ds, val_ds, test_ds, cfg.BATCH_SIZE, cfg.NUM_WORKERS
    )

    model = ResNet50TL(num_classes=cfg.NUM_CLASSES, dropout=0.2, pretrained=True).to(device)
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    scaler = GradScaler(enabled=cfg.MIXED_PRECISION)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(
        cfg.LOGS_DIR / "training_logs.csv",
        fieldnames=["phase","epoch","train_loss","val_loss","val_acc","val_ece","lr","time_s"],
    )

    best_acc, best_state = 0.0, None

    #Phase 1 (head-only)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=cfg.LR_HEAD, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.EPOCHS_HEAD))
    print(f"\nPhase 1: head-only for {cfg.EPOCHS_HEAD} epoch(s)")
    for epoch in range(1, cfg.EPOCHS_HEAD + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, clip=cfg.GRAD_CLIP_NORM)
        vl, va, ve, _, _ = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        logger.log({"phase":"head","epoch":epoch,"train_loss":f"{tr:.4f}",
                    "val_loss":f"{vl:.4f}","val_acc":f"{va:.4f}",
                    "val_ece":f"{ve:.4f}","lr":optimizer.param_groups[0]["lr"],
                    "time_s":int(time.time()-t0)})
        print(f"[Head {epoch}/{cfg.EPOCHS_HEAD}] val_acc={va:.4f} | val_ece={ve:.4f}")
        if va > best_acc:
            best_acc, best_state = va, model.state_dict()

    #Phase 2 (fine-tune)
    unfreeze_all(model)
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("backbone.fc") else backbone_params).append(p)
    optimizer = AdamW(
        [{"params": backbone_params, "lr": cfg.LR_FT},
         {"params": head_params,     "lr": cfg.LR_FT * 5.0}],
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.EPOCHS_FINETUNE))

    patience, wait = cfg.EARLY_STOP_PATIENCE, 0
    print(f"\nPhase 2: fine-tune up to {cfg.EPOCHS_FINETUNE} epoch(s) (patience {patience})")
    try:
        for epoch in range(1, cfg.EPOCHS_FINETUNE + 1):
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, clip=cfg.GRAD_CLIP_NORM)
            vl, va, ve, _, _ = evaluate(model, val_loader, device, criterion)
            scheduler.step()
            logger.log({"phase":"finetune","epoch":epoch,"train_loss":f"{tr:.4f}",
                        "val_loss":f"{vl:.4f}","val_acc":f"{va:.4f}",
                        "val_ece":f"{ve:.4f}","lr":optimizer.param_groups[0]["lr"],
                        "time_s":int(time.time()-t0)})
            print(f"[FT {epoch}/{cfg.EPOCHS_FINETUNE}] val_acc={va:.4f} | val_ece={ve:.4f}")
            if va > best_acc:
                best_acc, best_state, wait = va, model.state_dict(), 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break
    finally:
        logger.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    #Temperature scaling
    print("\nFitting temperature on validation set …")
    _, _, _, val_logits, val_labels = evaluate(model, val_loader, device, criterion)
    temp_scaler = TemperatureScaler(init_T=cfg.TEMPERATURE_INIT).to(device)
    try:
        temp_scaler.fit(val_logits, val_labels)
        T_value = temp_scaler.T.detach().cpu().item()
    except Exception as e:
        print(f"Calibration failed ({e}); defaulting T = 1.0")
        T_value = 1.0
    print(f"Fitted temperature T = {T_value:.3f}")

    #Checkpoint Saving
    torch.save({
        "model_state": model.state_dict(),
        "num_classes": cfg.NUM_CLASSES,
        "class_to_idx": train_ds.class_to_idx,
        "img_size": cfg.IMG_SIZE,
        "mean": cfg.MEAN,
        "std": cfg.STD,
        "temperature": T_value,
    }, args.out)
    print(f"\nSaved best model to {args.out} | best val acc={best_acc:.4f}")

    #Plotting
    subprocess.run([sys.executable, "-m", "scripts.plot_training"], check=False)

if __name__ == "__main__":
    main()