from torchvision import transforms


simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])


class SimCLRTransform:
    def __init__(self):
        self.transform = simclr_transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)

        return view1, view2