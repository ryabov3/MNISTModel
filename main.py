from model import MNISTModel
from dataset import train_loader, test_loader
from visualize import visualize_loss_in_epoch
from earlystopping import Earlystopping
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.001
save_path = "model/MNIST_best_model.pth"

model = MNISTModel(1, 10).to(device)
early_stopping = Earlystopping(attemps=5)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
loss_fn = nn.CrossEntropyLoss()

train_loss_values = []
val_loss_values = []
acc_values = []
EPOCH = 50

for num_epoch in range(1, EPOCH + 1):
    train_loss = 0

    model.train()
    for features, targets in tqdm(train_loader, desc=f"Epoch [{num_epoch}/{EPOCH}] | Train loader"):
        optimizer.zero_grad()
        features, targets = features.to(device), targets.to(device)

        y_pred = model(features)
        loss = loss_fn(y_pred, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    acc = 0
    with torch.no_grad():
        for features, targets in tqdm(test_loader, desc=f"Epoch [{num_epoch}/{EPOCH}] | Val loader"):
            features, targets = features.to(device), targets.to(device)
            
            y_pred = model(features)
            loss = loss_fn(y_pred, targets)
            val_loss += loss.item()
            predicted_classes = y_pred.argmax(dim=1)
            acc += (predicted_classes == targets).sum().item() / len(targets)
    val_loss /= len(test_loader)
    acc /= len(test_loader)
    
    train_loss_values.append(train_loss)
    val_loss_values.append(val_loss)
    acc_values.append(acc)
    
    print(f"\033[31mEpoch [{num_epoch}/{EPOCH}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {acc * 100:.2f}\033[0m")

    if early_stopping(val_loss):
        model_state = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": num_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "accuracy": round(acc * 100, 2),
            "val_loss": val_loss,
            "train_loss": train_loss
        }
        torch.save(model_state, save_path)
        print(F"\033[32m✅ Earlystopping на {num_epoch}. Модель сохранена в {save_path}\033[0m]")

visualize_loss_in_epoch(train_loss=train_loss_values, val_loss=val_loss_values)