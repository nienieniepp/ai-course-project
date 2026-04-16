"""Simple training script for FER-2013 emotion classification."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from dataset import create_dataloaders
from model import EmotionCNN
from utils import ensure_dir


DEFAULT_SAVE_PATH = "models/emotion_cnn.pth"
RESULTS_DIR = "results"


def train_one_epoch(model, dataloader, criterion, optimizer):
    """Run one training epoch and return average loss/accuracy."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def validate_one_epoch(model, dataloader, criterion):
    """Run one validation epoch and return average loss/accuracy."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def build_confusion_matrix(model, dataloader, num_classes: int):
    """Build confusion matrix on validation data."""
    model.eval()
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            for true_label, pred_label in zip(labels, predictions):
                matrix[int(true_label), int(pred_label)] += 1

    return matrix


def plot_training_curves(history, save_path: str):
    """Plot loss and accuracy curves and save figure."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix, class_names, save_path: str):
    """Plot confusion matrix heatmap and save figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    max_count = int(confusion_matrix.max().item()) if confusion_matrix.numel() else 0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[i, j].item())
            text_color = "white" if value > max_count / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a simple FER CNN model")
    parser.add_argument("--data_dir", default="data", help="Dataset root folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--save_path", default=DEFAULT_SAVE_PATH, help="Path to save best model"
    )
    args = parser.parse_args()

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    model = EmotionCNN(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    ensure_dir("models")
    ensure_dir(RESULTS_DIR)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to: {args.save_path}")

    plot_training_curves(history, str(Path(RESULTS_DIR) / "training_curves.png"))
    confusion_matrix = build_confusion_matrix(model, val_loader, num_classes=len(class_names))
    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        str(Path(RESULTS_DIR) / "confusion_matrix.png"),
    )

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    print("Saved training curves to: results/training_curves.png")
    print("Saved confusion matrix to: results/confusion_matrix.png")


if __name__ == "__main__":
    main()
