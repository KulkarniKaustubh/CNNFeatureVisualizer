import requests
import torch
import numpy as np
from torch import jit
import torch.nn as nn
from torchvision import models

import torchboard as tb


# Define the CNN model
class SmallCNN(nn.Module):
    def __init__(self, channels=3, height=255, width=255):
        super(SmallCNN, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height

        self.fc_dim = 0

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._get_fc_dim()

        self.fc1 = nn.Linear(self.fc_dim, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def _get_fc_dim(self):
        example_img = torch.randn(1, self.channels, self.width, self.height)

        with torch.no_grad():
            x = self.conv1(example_img)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = x.view(x.size(0), -1)

            self.fc_dim = x.size(1)

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

    tb.init(project_id="test project", model_class=SmallCNN)

    example_input = torch.randn(1, 3, 255, 255)
    output = model(example_input)

    # file_path = "temp.pt"
    # torch.jit.save(torch.jit.script(model), file_path)

    tb.visualize_convs(model)
    # x = tb.visualizers.CNNLayerVisualization(model, 0, 5)
    # x.visualise_layer_without_hooks()

    # model = models.vgg16(pretrained=True).features
    # layer_vis = tb.CNNLayerVisualization(model, 0, 5)

    # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()
    # layer_vis.visualise_layer_without_hooks()
