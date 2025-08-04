from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    #PATHS
    ROOT: Path = Path(__file__).resolve().parent
    DATA_ROOT: Path = ROOT / "data"
    RAW_DIR: Path = Path("/path/to/data/raw/wikiart")
    SUBSET_DIR: Path = Path("/path/to/data/wikiart_subset")
    TRAIN_DIR: Path = SUBSET_DIR / "train"
    VAL_DIR: Path = SUBSET_DIR / "val"
    TEST_DIR: Path = SUBSET_DIR / "test"
    CHECKPOINTS: Path = ROOT / "checkpoints"
    LOGS_DIR: Path = ROOT / "logs"
    PLOTS_DIR: Path = LOGS_DIR / "plots"
    CLASSES_CSV: Path = RAW_DIR / "classes.csv"

    #DATA
    NUM_CLASSES: int = 14
    IMG_SIZE: int = 256
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    #TRAIN
    EPOCHS_HEAD: int = 8
    EPOCHS_FINETUNE: int = 30     # longer FT
    LR_HEAD: float = 1e-3
    LR_FT: float = 2e-4           # try 5e-5 if val oscillates
    WEIGHT_DECAY: float = 1e-2  
    LABEL_SMOOTH: float = 0.00    # less smoothing -> higher peak confidences
    EARLY_STOP_PATIENCE: int = 8  # wait longer
    MIXED_PRECISION: bool = True
    GRAD_CLIP_NORM: float = 1.0

    # INFERENCE/CALIBRATION
    CONF_THRESH: float = 0.80
    TTA_N: int = 6                # used as a base; multi-scale TTA added in predict.py
    TEMPERATURE_INIT: float = 1.0

cfg = Config()