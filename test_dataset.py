import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("Dataset downloaded successfully!")
print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("Classes:", dataset.classes)