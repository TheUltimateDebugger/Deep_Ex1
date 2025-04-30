import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# class MLPClassifier(nn.Module):
#     def __init__(self):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(180, 180)
#         self.fc2 = nn.Linear(180, 180)
#         self.fc3 = nn.Linear(180, 7)  # 7 output neurons
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # flatten
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.output(x)


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return self.output(x)