"""Dataset loading utilities for FER-2013 style folders."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    """Transforms for grayscale FER images resized to 48x48."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def load_datasets(data_dir: str):
    """Load train/val ImageFolder datasets.

    Expected layout:
    data_dir/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Missing train/val folders under '{data_dir}'. "
            "Expected ImageFolder format: data/train and data/val."
        )

    transform = get_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Make sure train and val use the same class order.
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Class folders differ between train and val. "
            "Please keep both splits with identical class names."
        )

    return train_dataset, val_dataset


def create_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 0):
    """Create dataloaders and return class names."""
    train_dataset, val_dataset = load_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, train_dataset.classes
