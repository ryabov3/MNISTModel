import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from dataset import test_loader
from model import MNISTModel

checkpoint = torch.load("models/MNIST_model.pth")

model = MNISTModel(1, 10)
model.load_state_dict(checkpoint['model_state_dict'])

loss_fn = nn.CrossEntropyLoss()
test_loss = 0
accuracy = 0

model.eval()
with torch.no_grad():
    for features, target in tqdm(test_loader, desc="Test Loader"):
        predict = model(features)
        loss = loss_fn(predict, target)
        test_loss += loss.item()
        predicted_classes = predict.argmax(dim=1)
        accuracy += (predicted_classes == target).sum().item() / len(target)
    accuracy /= len(test_loader)
    test_loss /= len(test_loader)
logger.warning(f"Test Loss: {test_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")


