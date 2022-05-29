from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5,1,2)
        self.conv2 = nn.Conv2d(16, 32, 5,1,2)
        self.conv3 = nn.Conv2d(32, 64, 5,1,2)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.3)
        self.flatten = nn.Flatten(),
        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        # print(x.size())
        x = F.relu(self.conv1(x))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = F.relu(self.conv3(x))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = self.drop(x)
        # print(x.size())
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        # print(x)
        x = self.drop(x)
        # print(x)
        x = self.fc2(x)
        # print(x)
        return x


