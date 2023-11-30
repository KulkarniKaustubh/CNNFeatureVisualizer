import torch
import numpy as np
from torch import jit
import torch.nn as nn
from torchvision import models

# import sys
#
# sys.path.append("..")

import torchboard as tb


# Define the CNN model
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.LazyLinear(64)
        # self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.LazyLinear(10)
        # self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Create an instance of the model
    model = SmallCNN()

    tb.visualize_convs(model)

    # model = models.vgg16(pretrained=True).features
    # layer_vis = tb.CNNLayerVisualization(model, 0, 5)

    # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()
    # layer_vis.visualise_layer_without_hooks()
