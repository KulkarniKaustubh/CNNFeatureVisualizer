import torch
import torch.nn as nn


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
