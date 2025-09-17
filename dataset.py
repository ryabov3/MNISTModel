from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2

BATCH_SIZE = 16

transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(28, 28))
])

train_dataset = MNIST(root="mnist/train", download=False, train=True, transform=transforms)
test_dataset = MNIST(root="mnist/test", download=False, train=False, transform=transforms)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)