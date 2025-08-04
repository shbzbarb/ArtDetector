import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
import os
from models.resnet50_transfer import ResNet50TL

MODEL_PATH = '/path/to/checkpoint/best_art_style_classifier.pth'
CLASS_FILE = './ArtClassifier/class_names.json'

CLASS_NAMES = [
    "abstract", "animal-painting", "cityscape", "figurative", "flower-painting",
    "genre-painting", "landscape", "marina", "mythological-painting",
    "nude-painting-nu", "portrait", "religious-painting",
    "still-life", "symbolic-painting"
]

if os.path.exists(CLASS_FILE):
    with open(CLASS_FILE, "r") as f:
        CLASS_NAMES = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

num_classes = len(CLASS_NAMES)
model = ResNet50TL(num_classes=num_classes)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

def predict(input_image: Image.Image) -> dict:
    if input_image is None:
        return {}
    
    processed_image = transform(input_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(processed_image)
        probabilities = F.softmax(logits, dim=1)[0]
    
    return {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Artwork"),
    outputs=gr.Label(num_top_classes=3, label="Art Style Prediction"),
    title="Art Style Detector",
    description="Upload an image to classify its art style with a ResNet50 model. The top 3 predictions are shown",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()