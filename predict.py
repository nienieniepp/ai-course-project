"""Single image prediction script and reusable prediction function."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.nn.functional import softmax

from dataset import get_transforms
from model import EmotionCNN
from utils import EMOTION_LABELS, format_probabilities


MODEL_PATH = "models/emotion_cnn.pth"


def load_model(model_path: str = MODEL_PATH):
    """Load trained model weights for inference."""
    model = EmotionCNN(num_classes=len(EMOTION_LABELS))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(image_path: str, model_path: str = MODEL_PATH):
    """Predict emotion and class probabilities for one image."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    image = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    tensor = transform(image).unsqueeze(0)

    model = load_model(model_path)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = softmax(logits, dim=1).squeeze(0).tolist()

    predicted_index = int(torch.argmax(torch.tensor(probabilities)).item())
    predicted_label = EMOTION_LABELS[predicted_index]

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
