import torch
import torchvision
import torchvision.transforms as transforms

from simclr_model import SimCLR


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load trained model
model = SimCLR(projection_dim=128)

model.load_state_dict(
    torch.load(
        "simclr_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# CIFAR-10 dataset
transform = transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)


# Take one image
image, label = dataset[0]

image = image.unsqueeze(0).to(device)


# Extract feature
with torch.no_grad():

    feature, projection = model(image)


print("Image class:", dataset.classes[label])
print("Feature shape:", feature.shape)
print("Projection shape:", projection.shape)
print("Feature extracted successfully!")