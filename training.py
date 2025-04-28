from torch.utils.data import Dataset, DataLoader
import mlp_class
import torch
import torch.nn as nn
import torch.optim as optim


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dummy dataset
X_train = torch.randint(0, 2, (100, 180)).float()
# Labels: integers between 0 and 6
y_train = torch.randint(0, 7, (100,))

# Create Dataset
dataset = CustomDataset(X_train, y_train)

# Create DataLoader
loader = DataLoader(dataset, batch_size=10, shuffle=True)
# Model
model = mlp_class.MLPClassifier()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training parameters
num_epochs = 20
batch_size = 10


for epoch in range(num_epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
