import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import cfg
from tqdm import tqdm

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def list_categories(raw_root: Path) -> List[str]:
    cats = [d.name for d in raw_root.iterdir() if d.is_dir()]
    cats.sort()
    return cats

def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def allocate_counts(avail: Dict[str, int], total_target: int) -> Dict[str, int]:
    cats = sorted(avail.keys())
    n = len(cats)
    base = total_target // n
    rem = total_target % n

    want = {c: base + (i < rem) for i, c in enumerate(cats)}
    for c in cats:
        want[c] = min(want[c], avail[c])

    current = sum(want.values())
    if current < total_target:
        deficit = total_target - current
        i = 0
        while deficit > 0:
            c = cats[i % n]
            headroom = avail[c] - want[c]
            if headroom > 0:
                add = min(headroom, deficit)
                want[c] += add
                deficit -= add
            i += 1
        current = sum(want.values())

    if current < total_target:
        tqdm.write(f"Warning: only {current} images available across classes; will build a {current}-image subset")
    return want

def split_counts(k: int) -> Tuple[int,int,int]:
    t = int(k * 0.60)
    v = int(k * 0.20)
    s = k - t - v
    return t, v, s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=6800, help="Total images across all classes")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clean", action="store_true", help="Delete existing subset before creating")
    args = parser.parse_args()

    random.seed(args.seed)

    raw_root = cfg.RAW_DIR
    out_root = cfg.SUBSET_DIR
    assert raw_root.exists(), f"Raw root not found: {raw_root}"

    cats = list_categories(raw_root)
    if len(cats) != 14:
        tqdm.write(f"Note: found {len(cats)} categories. Proceeding with these:\n{cats}")

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    for split in ["train", "val", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    files_by_cat: Dict[str, List[Path]] = {}
    for c in tqdm(cats, desc="Scanning classes"):
        src_dir = raw_root / c
        imgs = list_images(src_dir)
        if not imgs:
            tqdm.write(f"Warning: no images in {src_dir}")
        random.shuffle(imgs)
        files_by_cat[c] = imgs

    avail = {c: len(files_by_cat[c]) for c in cats}
    want = allocate_counts(avail, args.total)

    for split in ["train", "val", "test"]:
        for c in cats:
            (out_root / split / c).mkdir(parents=True, exist_ok=True)


    CopyTask = Tuple[Path, Path, str]
    tasks: List[CopyTask] = []
    copied = {"train": 0, "val": 0, "test": 0}

    for c in cats:
        k = want[c]
        if k == 0:
            continue
        imgs = files_by_cat[c][:k]
        n_train, n_val, n_test = split_counts(k)
        assignments = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:n_train+n_val+n_test],
        }
        for split, paths in assignments.items():
            dst_dir = out_root / split / c
            for src in paths:
                tasks.append((src, dst_dir / src.name, split))

    for src, dst, split in tqdm(tasks, desc="Copying subset", unit="img"):
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied[split] += 1

    total_copied = sum(copied.values())
    print("Subset created at:", out_root)
    print(f"Counts: train={copied['train']} ({copied['train']/max(1,total_copied):.1%})  "
          f"val={copied['val']} ({copied['val']/max(1,total_copied):.1%})  "
          f"test={copied['test']} ({copied['test']/max(1,total_copied):.1%})")
    print("Per-class targets vs availability:")
    for c in cats:
        print(f" - {c}: want={want[c]}  avail={avail[c]}")

if __name__ == "__main__":
    main()