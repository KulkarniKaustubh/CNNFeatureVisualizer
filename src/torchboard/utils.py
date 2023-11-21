import torch
import torch.nn as nn


def get_module(model: nn.Module, idx: int) -> nn.Module:
    counter = 0
    if isinstance(idx, int) and idx >= 0:
        for module in model.modules():
            if counter == idx:
                return module

            counter += 1

    raise TypeError(f"Invalid index {idx}")
