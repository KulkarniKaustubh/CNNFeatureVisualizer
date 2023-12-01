import torch.nn as nn
import requests
import torch
import os
import json

REST = os.getenv("REST") or "localhost:5000"


def send_model(model, reqmethod, endpoint, debug=True):
    file_path = "temp.pt"
    torch.jit.save(torch.jit.script(model), file_path)
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
    response = reqmethod(
        f"http://{REST}/{endpoint}", files=files, data=payload
    )
    os.remove(file_path)

    if response.status_code == 200:
        jsonResponse = json.dumps(response.json(), indent=4, sort_keys=True)
        print(jsonResponse)
        return
    else:
        print(
            f"response code is {response.status_code}, raw response is {response.text}"
        )
        return response.text


def visualize_convs(model: nn.Module) -> None:
    send_model(model, requests.post, "visualize")
