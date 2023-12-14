import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_graph(
    x_values: list, y_values: list, title: str, x_label: str, y_label: str
):
    fig, ax = plt.subplots()

    ax.plot(x_values, y_values)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.legend()

    return fig


def load_model_weights(model: nn.Module, weights_file: str) -> nn.Module:
    layer_weights = torch.load(weights_file)

    print("Received model is: ", model)

    # Load each layer's weights back into the model
    for name, param in model.named_parameters():
        param.data.copy_(layer_weights[name])

    return model


def get_conv_layer_idx_dict(model: nn.Module) -> dict:
    """
    Method to obtain conv layer information given an nn.Module.

    Arguments
    ---------
    model: the model from which to obtain conv layer information

    Returns
    -------
    dict: contains the conv layer information
        keys -> conv layer indices, values -> output channels
    """
    conv_layer_idx_dict = {}

    for i, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            conv_layer_idx_dict[i] = module.out_channels

    return conv_layer_idx_dict


def get_from_module(selected_module: nn.Module, idx: int) -> nn.Module:
    """
    Method to get a specific sub-module from a pytorch model or module.

    Arguments
    ---------
    selected_module: nn.Module
        The module to iterate over and get based on `idx`.

    idx: int
        The index of the module to return.

    Returns
    -------
    nn.Module: The module that is requested using an index.
    """
    counter = 0
    module_list = list(selected_module.modules())

    if isinstance(idx, int) and idx >= 0:
        # The [1:] is to exclude the entire module itself.
        for selected_module in module_list[1:]:
            if counter == idx:
                return selected_module

            counter += 1

    raise TypeError(f"Invalid index {idx}")
