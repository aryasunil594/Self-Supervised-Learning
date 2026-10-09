import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from augmentations import SimCLRTransform
from simclr_model import SimCLR
from contrastive_loss import NTXentLoss

from tqdm import tqdm


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Dataset
# -----------------------------

base_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)


class ContrastiveDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, _ = self.dataset[index]

        view1, view2 = self.transform(image)

        return view1, view2


dataset = ContrastiveDataset(base_dataset)


# -----------------------------
# DataLoader
# -----------------------------

batch_size = 128

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)


# -----------------------------
# Model
# -----------------------------

model = SimCLR(
    projection_dim=128
)

model = model.to(device)


# -----------------------------
# Loss
# -----------------------------

criterion = NTXentLoss(
    temperature=0.5
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# Training
# -----------------------------

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{epochs}"
    )

    for view1, view2 in progress_bar:

        view1 = view1.to(device)
        view2 = view2.to(device)

        # Forward pass
        _, z1 = model(view1)
        _, z2 = model(view2)

        # Contrastive loss
        loss = criterion(z1, z2)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Average Loss: {average_loss:.4f}"
    )


# -----------------------------
# Save trained model
# -----------------------------

torch.save(
    model.state_dict(),
    "simclr_model.pth"
)

print("\nTraining completed!")
print("Model saved as: simclr_model.pth")