import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):

    def __init__(self, projection_dim=128):
        super().__init__()

        # ResNet-18 encoder
        self.encoder = models.resnet18(weights=None)

        # Remove the original classification layer
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):

        # Image -> feature representation
        h = self.encoder(x)

        # Feature -> projection representation
        z = self.projector(h)

        return h, z