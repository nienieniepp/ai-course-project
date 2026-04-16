"""Single-image inference utilities and CLI."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import get_transforms
from model import EmotionCNN
from utils import EMOTION_LABELS, format_probabilities


MODEL_PATH = "models/emotion_cnn.pth"


def load_model(model_path: str = MODEL_PATH):
    """Load trained model weights and return a ready-to-use model."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = EmotionCNN(num_classes=len(EMOTION_LABELS))
    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: str):
    """Load one image and convert it to a model input tensor."""
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_file).convert("RGB")
    tensor = get_transforms()(image).unsqueeze(0)
    return tensor


def predict_image(image_path: str, model_path: str = MODEL_PATH):
    """Predict a label and class probabilities for a single image."""
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0)

    predicted_index = int(torch.argmax(probabilities_tensor).item())
    predicted_label = EMOTION_LABELS[predicted_index]
    probabilities = probabilities_tensor.tolist()

    return {
        "predicted_label": predicted_label,
        "probabilities": format_probabilities(EMOTION_LABELS, probabilities),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict emotion for one face image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model", default=MODEL_PATH, help="Path to trained model weights"
    )
    args = parser.parse_args()

    result = predict_image(args.image, args.model)
    print(f"Predicted emotion: {result['predicted_label']}")
    print("Class probabilities (%):")
    for item in result["probabilities"]:
        print(f"- {item['label']}: {item['probability']}")


if __name__ == "__main__":
    main()
