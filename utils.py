"""Utility helpers used by training, prediction, and Flask app."""

from pathlib import Path
import uuid

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def format_probabilities(labels, probabilities):
    """Pair labels with probabilities as rounded percentages."""
    return [
        {"label": label, "probability": round(float(prob) * 100, 2)}
        for label, prob in zip(labels, probabilities)
    ]


def save_uploaded_file(file_storage, upload_dir: str = "uploads") -> str:
    """Save uploaded file with a unique name and return its path."""
    ensure_dir(upload_dir)
    extension = Path(file_storage.filename).suffix.lower()
    filename = f"{uuid.uuid4().hex}{extension}"
    save_path = Path(upload_dir) / filename
    file_storage.save(save_path)
    return str(save_path)
