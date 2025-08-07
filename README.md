# ArtClassifier: Deep Learning Model for Art Style Detection

This repository contains the implementation of ArtClassifier, a deep learning-based system for classifying paintings into one of 14 distinct art styles. The project leverages a fine-tuned ResNet-50 model trained on the [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart) and provides a complete end-to-end pipeline for data preparation, training, evaluation, and inference(or prediction). A live demo is available on [Hugging Face Spaces](https://huggingface.co/spaces/ahmdshbz/ArtDetector).

## Key Features

- **High-Performance Model**: Built upon a ResNet-50 architecture, pre-trained on ImageNet and fine-tuned for maximum accuracy.

- **Two-Phase Training Strategy**:
  - Initially the model trains only the new classification head to quickly adapt to the dataset.
  - Then the model fine-tunes the entire network at a lower learning rate for optimal performance.

- **Confidence Calibration**: Implements post-hoc `Temperature Scaling` to calibrate output probabilities, ensuring reliable confidence scores.

- **Robust Inference**: Employs Test-Time Augmentation (TTA) during prediction for improved stability and accuracy.

- **Comprehensive Evaluation**: Generates detailed performance metrics, including accuracy, macro F1-score, Expected Calibration Error (ECE), confusion matrices, and ROC curves with ROC AUC scores for each art style.

- **Interactive Web App**: A user-friendly **Gradio application** allows easy, real-time classification of artwork images.

- **Reproducible Environment**: Managed via a Conda `environment.yml` file ensuring straightforward setup and complete reproducibility.


## Model Architecture

The core model, **ResNet50TL**, is adapted from torchvision’s ResNet-50:

- Initialized with pre-trained ImageNet (IMAGENET1K\_V2) weights.
- Replaces the original 1000-class fully-connected layer with a new head containing:
  - Dropout (for regularization)
  - Final linear layer producing logits for the 14 classes
- Training involves initially freezing the backbone (`freeze_backbone`) and subsequently fine-tuning the entire network (`unfreeze_all`).


### Prerequisites

- Git
- Conda
- NVIDIA GPU with CUDA support

```bash
git clone https://github.com/your-username/ArtClassifier.git
```

```bash
cd ArtClassifier
```

### Create and Activate Conda Environment

This setup uses Python 3.12 and PyTorch with CUDA 11.8.

```bash
conda env create -f environment.yml
```

```bash
conda activate art_guide_gradio
```

### Download the Dataset

Download the WikiArt dataset from [Kaggle](https://www.kaggle.com/datasets/steubk/wikiart). Extract and ensure the resulting directory structure is:

```
data/wikiart/
```

Alternatively, update the dataset location in `config.py`:

```python
# config.py
@dataclass
class Config:
    RAW_DIR: Path = Path("/path/to/your/data/wikiart") # EDIT THIS LINE
    SUBSET_DIR: Path = Path("./data/wikiart_subset")  # Automatically created
```

### Prepare Data Subset

Run the following script from the project root to create balanced subsets for training, validation, and testing:

```bash
python scripts/data_preparation.py
```


## Usage Workflow

### 1. Train the Model

Begin model training, including both training phases and automated logging:

```bash
python scripts/train.py
```

- Best checkpoint saved to `checkpoints/best_art_style_classifier.pth`
- Training logs stored in `logs/training_logs.csv`
- Loss and accuracy plots saved in project root

### 2. Evaluate the Model

Evaluate model performance on the test set:

```bash
python scripts/evaluate.py
```

Outputs:

- Metrics printed to the console
- Visualizations (`confusion_matrix.png`, `roc_curve.png`) saved in project root

### 3. Make Predictions from Command Line

Classify individual or multiple images:

## Single image inference
```bash
python scripts/predict.py /path/to/artwork.jpg 
```
## Folder with multiple images inference
```bash
python scripts/predict.py /path/to/art_folder/
```

### 4. Launch the Interactive Web App

Update the model path in `app.py`:

```python
# app.py
MODEL_PATH = './checkpoints/best_art_style_classifier.pth'
```

Run the app:

```bash
python app.py
```

Open your browser at [http://127.0.0.1:7860](http://127.0.0.1:7860) to use the interface.
