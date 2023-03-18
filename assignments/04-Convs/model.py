import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=32, kernel_size=5, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.flatten = nn.Flatten(1, 3)
        self.linear1 = nn.Linear(3136, 256, True)
        self.linear2 = nn.Linear(256, 128, True)
        self.linear3 = nn.Linear(128, num_classes, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        x = self.ReLU(x)
        x = self.linear3(x)
        return x
