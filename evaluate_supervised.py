import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(
    torch.load("supervised_model_full.pth", map_location=device)
)

model = model.to(device)
model.eval()

actual = []
predicted = []

print("Evaluating full model...", flush=True)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.numpy())
        predicted.extend(predictions.cpu().numpy())

print("\nSupervised Model Results")
print("Accuracy:", round(accuracy_score(actual, predicted) * 100, 2), "%")
print("Precision:", round(precision_score(actual, predicted, average="macro", zero_division=0), 4))
print("Recall:", round(recall_score(actual, predicted, average="macro", zero_division=0), 4))
print("F1-score:", round(f1_score(actual, predicted, average="macro", zero_division=0), 4))