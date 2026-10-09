import torchvision
import matplotlib.pyplot as plt

from augmentations import SimCLRTransform


transform = SimCLRTransform()

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False
)

image, label = dataset[0]

view1, view2 = transform(image)

print("Original image size:", image.size)
print("View 1 shape:", view1.shape)
print("View 2 shape:", view2.shape)
print("Original label:", dataset.classes[label])


plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.imshow(view1.permute(1, 2, 0))
plt.title("Augmented View 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(view2.permute(1, 2, 0))
plt.title("Augmented View 2")
plt.axis("off")

plt.tight_layout()
plt.show()