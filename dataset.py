"""Dataset helpers for FER-2013 style ImageFolder directories."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    """Return basic transforms for 48x48 grayscale FER images."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def create_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 0):
    """Create train and validation dataloaders from ImageFolder structure.

    Expected structure:
    data_dir/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected data/train and data/val folders in ImageFolder format."
        )

    transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, train_dataset.classes
