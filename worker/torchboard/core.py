from typing import Type
import torch.nn as nn
import torch
import os
import json
import requests
import inspect

import torchboard.rest as rest


def _get_model_source_code(model_class: Type) -> str:
    source_code_file_path = "temp.py"
    with open(source_code_file_path, "w") as f:
        f.write(inspect.getsource(model_class))
    with open(source_code_file_path, "r") as f:
        model_source_code = f.read()

    return model_source_code


def init(project_id: str, model_class: Type, **model_class_args) -> None:
    endpoint = "initialize"

    data = {
        "project_id": project_id,
        "model_class_name": model_class.__name__,
        "model_source_code": _get_model_source_code(model_class),
        "model_class_args": {**model_class_args},
    }
    payload = {"data": json.dumps(data)}

    rest._request_response(endpoint, requests.post, payload)


def _send_model(model) -> None:
    endpoint = "visualize2"

    file_path = "layer_weights.pth"
    layer_weights = {}
    for name, param in model.named_parameters():
        layer_weights[name] = param.clone().detach().cpu()

    # Save the dictionary containing layer weights
    torch.save(layer_weights, file_path)

    # Define additional fields
    username = "example_user"
    model_name = "example_model"
    iteration_number = 1

    # Create a dictionary with your fields
    data = {
        "username": username,
        "modelname": model_name,
        "iteration_number": iteration_number,
    }
    files = {"file": open(file_path, "rb")}
    payload = {"data": json.dumps(data)}

    rest._request_response(endpoint, requests.post, payload, files)

    os.remove(file_path)


def visualize_convs(model: nn.Module) -> None:
    _send_model(model)


def log(metric_dict: dict) -> None:
    """
    Method to log training metrics.

    Arguments
    ---------
    metric_dict : dictionary of metrics to track
        the keys can be from the following:
            epoch
            train-loss
            train-acc
            val-loss
            val-acc
            test-loss
            test-acc
    """
    endpoint = "logs"

    supported_metrics = [
        "epoch",
        "train-loss",
        "train-acc",
        "val-loss",
        "val-acc",
        "test-loss",
        "test-acc",
    ]

    for metric in metric_dict.keys():
        assert (
            metric in supported_metrics
        ), f"{metric} logging is not supported."

    data = {**metric_dict}

    rest._request_response(endpoint, requests.post, data)
