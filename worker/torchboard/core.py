from typing import Type
import torch.nn as nn
import torch
import os
import json
import requests
import hashlib
import inspect
import time
import random

import torchboard.rest as rest


# re-initialized after every tb.init() call, so the model
# hash changes for every run, maintaining the username
_username = None
_project_id = None


def _get_model_source_code(model_class: Type) -> str:
    source_code_file_path = "temp.py"
    with open(source_code_file_path, "w") as f:
        f.write(inspect.getsource(model_class))
    with open(source_code_file_path, "r") as f:
        model_source_code = f.read()

    return model_source_code


def _generate_model_hash() -> str:
    data = f"{time.time()}{random.random()}"

    # Generate SHA-1 hash
    sha1_hash = hashlib.sha1(data.encode()).hexdigest()

    # Truncate to 12 characters
    truncated_hash = sha1_hash[:12]

    return truncated_hash


def init(
    username: str, project_id: str, model_class: Type, **model_class_args
) -> None:
    global _username, _project_id

    endpoint = "initialize"

    project_id = project_id.strip().replace(" ", "-")
    project_id += f"-{_generate_model_hash()}"

    _username = username
    _project_id = project_id

    data = {
        "username": username,
        "project_id": project_id,
        "model_class_name": model_class.__name__,
        "model_source_code": _get_model_source_code(model_class),
        "model_class_args": {**model_class_args},
    }
    payload = {"data": json.dumps(data)}

    rest._request_response(endpoint, requests.post, payload)

    postgres_endpoint = "postgres/insertRows"

    user_json_data = {"table_name": "users", "rows": [[username]]}
    rest._request_response(
        endpoint=postgres_endpoint,
        req_method=requests.post,
        data=None,
        files=None,
        json=user_json_data,
    )

    model_hash_json_data = {
        "table_name": "model_hashes",
        "rows": [[username, project_id]],
    }
    rest._request_response(
        endpoint=postgres_endpoint,
        req_method=requests.post,
        data=None,
        files=None,
        json=model_hash_json_data,
    )


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
    global _username, _project_id

    endpoint = "postgres/insertRows"

    supported_metrics = [
        "epoch",
        "train-loss",
        "train-acc",
        "val-loss",
        "val-acc",
        "test-loss",
        "test-acc",
    ]

    row_list = [_username, _project_id]
    for metric in supported_metrics:
        row_list.append(metric_dict.get(metric, None))

    log_json_data = {
        "table_name": "training_metrics",
        "rows": [row_list],
    }
    rest._request_response(
        endpoint=endpoint,
        req_method=requests.post,
        data=None,
        files=None,
        json=log_json_data,
    )
