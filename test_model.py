import torch

from simclr_model import SimCLR


# Create SimCLR model
model = SimCLR()

# Create a dummy batch of images
images = torch.randn(4, 3, 32, 32)

# Pass images through the model
features, projections = model(images)

print("Input shape:", images.shape)
print("Feature shape:", features.shape)
print("Projection shape:", projections.shape)