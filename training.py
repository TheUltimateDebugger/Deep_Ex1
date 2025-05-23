import mlp_class
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

from dataset import PeptideDataset


full_dataset = PeptideDataset("ex1 data\\ex1 data")

# 90% train, 10% test split
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Model, loss, optimizer
model = mlp_class.MLPBetterClassifier()
# Collect training labels only
train_labels = torch.tensor([full_dataset[i][1] for i in train_dataset.indices])
class_counts = torch.bincount(train_labels)
class_weights = 1.0 / class_counts.float()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 75
test_losses = []
train_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.float()
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)


    # Testing (Evaluation Phase)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():  # No need to calculate gradients during evaluation
        for test_x, test_y in test_loader:  # assuming you have a test_loader
            test_x = test_x.float()
            outputs = model(test_x)
            loss = criterion(outputs, test_y)
            test_loss += loss.item() * test_x.size(0)  # accumulate the loss

    test_loss /= len(test_loader.dataset)  # average test loss over the whole test set
    test_losses.append(test_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, marker='s', label='Test Loss')
plt.axvline(x=30, color='red', linestyle='--', label='Suggested Stop (Epoch 30)')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='upper right', frameon=True)
plt.grid(True)
plt.ylim(bottom=0)
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for test_x, test_y in test_loader:
        test_x = test_x.float()
        outputs = model(test_x)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# saves the model
torch.save(model.state_dict(), "trained_model.pth")
