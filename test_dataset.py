import torchvision
import torchvision.transforms as transforms

print("Loading CIFAR-10 dataset...")

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("\nDataset loaded successfully!")
print("Number of images:", len(train_dataset))
print("Number of classes:", len(train_dataset.classes))
print("Classes:", train_dataset.classes)

image, label = train_dataset[0]

print("\nSample image shape:", image.shape)
print("Sample label:", label)
print("Sample class:", train_dataset.classes[label])