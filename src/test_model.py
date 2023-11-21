import torch
import numpy as np
from torch import jit
import torch.nn as nn
from torchvision import models

from cnn_layer_visualization import CNNLayerVisualization
from misc_functions import preprocess_image, recreate_image, save_image
import pickle
import inspect

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

    def __getitem__(self, key):
        counter = 0
        if isinstance(key, int) and key >= 0:
            for module in self.modules():
                if counter == key:
                    return module

                counter += 1

        raise TypeError(f"Invalid key {key}")

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

    # print(inspect.getsource(SmallCNN))
    # print(model[1])

    # jit.save(jit.script(model), "../models/test_model.pth")

    # Save the model weights
    # torch.save(model.state_dict(), "../models/test_model.pth")

    # pretrained_model = models.vgg16(pretrained=True)
    # print(pretrained_model)
    # print(
    #     [
    #         name
    #         for name, _ in getattr(pretrained_model, "features").named_children()
    #     ]
    # )
    # for c in pretrained_model.children():
    #     print(c)
    #
    # layer_vis = tb.CNNLayerVisualization(pretrained_model, 2, 5)
    # layer_vis = tb.CNNLayerVisualization(model, 3, 5)

    # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()
    # layer_vis.visualise_layer_without_hooks()
