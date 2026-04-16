"""Simple CNN model for facial emotion recognition."""

import torch.nn as nn


class EmotionCNN(nn.Module):
    """Small and clear CNN baseline for 48x48 grayscale images."""

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 1 x 48 x 48
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 16 x 24 x 24
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32 x 12 x 12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 64 x 6 x 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
