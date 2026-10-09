
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset
from simclr_model import SimCLR

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Load trained SimCLR model
model = SimCLR(projection_dim=128)
model.load_state_dict(
    torch.load("simclr_model.pth", map_location=device)
)
model = model.to(device)
model.eval()

# Freeze the encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Load CIFAR-10 datasets
transform = transforms.ToTensor()

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False,
    transform=transform
)

test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False,
    transform=transform
)

def extract_features(dataset):
    loader = DataLoader(
        dataset, batch_size=256,
        shuffle=False, num_workers=0
    )

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.encoder(images)

            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)

print("Extracting training features...")
X_train, y_train = extract_features(train_data)

print("Extracting test features...")
X_test, y_test = extract_features(test_data)

print("Training feature shape:", X_train.shape)
print("Test feature shape:", X_test.shape)

# Train a linear classifier
classifier = nn.Linear(512, 10).to(device)

optimizer = torch.optim.Adam(
    classifier.parameters(), lr=0.001
)

criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256, shuffle=True
)

print("Training linear classifier...")

for epoch in range(10):
    classifier.train()
    total_loss = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/10, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )

# Evaluate
classifier.eval()
correct = 0
total = 0

with torch.no_grad():
    for start in range(0, len(X_test), 256):
        features = X_test[start:start + 256].to(device)
        labels = y_test[start:start + 256].to(device)

        outputs = classifier(features)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Linear Probe Accuracy: {accuracy:.2f}%")

torch.save(
    classifier.state_dict(),
    "linear_classifier.pth"
)

print("Classifier saved as linear_classifier.pth")
