
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from simclr_model import SimCLR

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load trained encoder and linear classifier
model = SimCLR(projection_dim=128)
model.load_state_dict(
    torch.load("simclr_model.pth", map_location=device)
)
model = model.to(device)
model.eval()

classifier = nn.Linear(512, 10)
classifier.load_state_dict(
    torch.load("linear_classifier.pth", map_location=device)
)
classifier = classifier.to(device)
classifier.eval()

# Test dataset
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

loader = DataLoader(
    dataset, batch_size=256,
    shuffle=False, num_workers=0
)

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        features = model.encoder(images)
        outputs = classifier(features)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Metrics
print("Accuracy:", round(
    accuracy_score(all_labels, all_predictions) * 100, 2
), "%")

print("Precision:", round(
    precision_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    ), 4
))

print("Recall:", round(
    recall_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    ), 4
))

print("F1-score:", round(
    f1_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    ), 4
))

print("\nClassification Report:")
print(classification_report(
    all_labels, all_predictions,
    target_names=dataset.classes,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes
)

display.plot(xticks_rotation=45)
plt.title("SimCLR Linear Probe - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion matrix saved as confusion_matrix.png")
