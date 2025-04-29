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
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(180, 180)
        self.fc2 = nn.Linear(180, 180)
        self.fc3 = nn.Linear(180, 180)
        self.output = nn.Linear(180, 7)  # 7 output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x

