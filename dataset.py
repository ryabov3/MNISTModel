import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

BATCH_SIZE = 16

transforms = v2.Compose([
    v2.Grayscale(),
    v2.ToTensor(),
    v2.Resize(size=(28, 28))
])

train_dataset = MNIST(root="mnist/train", download=False, train=True, transform=transforms)
test_dataset = MNIST(root="mnist/test", download=False, train=False, transform=transforms)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
