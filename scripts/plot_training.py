"""
Read logs/training_logs.csv and plot:
- train vs val loss
- val accuracy

Run:
    python -m scripts.plot_training
"""
import os, sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import cfg

def main():
    csv_path = cfg.LOGS_DIR / "training_logs.csv"
    assert csv_path.exists(), f"Log file not found: {csv_path}"
    df = pd.read_csv(csv_path)

    for col in ["epoch", "train_loss", "val_loss", "val_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    df["global_epoch"] = df.index + 1

    #Plot losses
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["global_epoch"], df["train_loss"], label="train_loss")
    ax.plot(df["global_epoch"], df["val_loss"], label="val_loss")
    ax.set_xlabel("Step (epochs in order logged)")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    out1 = cfg.PLOTS_DIR / "loss_curves.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    #Plot val accuracy
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["global_epoch"], df["val_acc"], label="val_acc")
    ax.set_xlabel("Step (epochs in order logged)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    out2 = cfg.PLOTS_DIR / "val_accuracy.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

if __name__ == "__main__":
    main()
