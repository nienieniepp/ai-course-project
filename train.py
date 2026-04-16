"""Simple training script for FER-2013 emotion classification."""

import argparse

import torch
from torch import nn, optim

from dataset import create_dataloaders
from model import EmotionCNN
from utils import ensure_dir


DEFAULT_SAVE_PATH = "models/emotion_cnn.pth"


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

    best_val_acc = 0.0
    ensure_dir("models")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to: {args.save_path}")

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
