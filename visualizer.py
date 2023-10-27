import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
from torchvision import utils


def visualize_conv(tensor, ch=0, all_kernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)

    test_layer = conv_layers[0]
    data = test_layer.weight.data.clone()

    print(data.shape)  # shape is (n, c, h, w)

    visTensor(data, ch=0, allkernels=False)

    plt.axis("off")
    plt.ioff()
    plt.show()

    # for i, conv_layer in enumerate(conv_layers):
    #     print(f"Conv layer {i+1}:")
    #     print(conv_layer)
    #     print()
