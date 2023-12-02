import torch
import torch.nn as nn
import torch.optim as optim
import sys
import io
import json, jsonpickle
import os
import requests
import base64
import inspect
from typing import Callable
from typing import Type
REST = os.getenv("REST") or "localhost:80"


def request_response(
    endpoint: str,
    req_method: Callable[..., requests.Response],
    data: dict,
    files: dict = None,
    debug: bool = True,
) -> None:
    response = req_method(f"http://{REST}/{endpoint}", data=data, files=files)

    if response.status_code == 200:
        jsonResponse = json.dumps(response.json(), indent=4, sort_keys=True)
        if debug:
            print(jsonResponse)
    else:
        print(
            f"response code: {response.status_code}, raw response: {response.text}"
        )

    if debug:
        print(response.text)

    return

# How to fix the problem when we have arguments in the init function
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


def get_model():
    model = SmallCNN()
    print("Model is: ")
    print(model)
    output = get_model_output(model=model)
    print("Model output: ")
    print(output)
    return model


def get_model_output(model):
    example_input = torch.randn(1, 3, 255, 255)
    output = model(example_input)
    return output


def sendmodel(reqmethod, endpoint, debug=True):
    model = get_model()
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
        "iterationNumber": iteration_number,
    }
    files = {"file": open(file_path, "rb")}
    payload = {"data": json.dumps(data)}
    # source_code_file_path='temp.py'
    # weights_file_path='temp_weights.pth'
    # with open(source_code_file_path, 'w') as f:
    #     f.write(inspect.getsource(SimpleModel))
    # torch.save(model, weights_file_path)
    # with open(source_code_file_path, 'r') as f:
    #     model_source_code = f.read()
    # state_dict_buffer = io.BytesIO()
    # torch.save(model.state_dict(), state_dict_buffer)

    # # files = {'weights': open(weights_file_path, 'rb')}
    # files = {'weights': state_dict_buffer}
    # data = {'source_code': model_source_code}
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
    
def _get_model_source_code(model_class: Type) -> str:
    source_code_file_path = "temp.py"
    with open(source_code_file_path, "w") as f:
        f.write(inspect.getsource(model_class))
    with open(source_code_file_path, "r") as f:
        model_source_code = f.read()

    return model_source_code
def init(project: str, model_class: Type, **model_class_args) -> None:
    endpoint = "initialize"

    data = {
        "project": project,
        "model_class_name": model_class.__name__,
        "model_source_code": _get_model_source_code(model_class),
        "model_class_args": {**model_class_args},
    }
    payload = {"data": json.dumps(data)}

    request_response(endpoint, requests.post, payload)

def send_model(model) -> None:
    endpoint = "visualize2"

    file_path = "layer_weights.pth"
    layer_weights = {}
    for name, param in model.named_parameters():
        layer_weights[name] = param.clone().detach().cpu()

    # Save the dictionary containing layer weights
    torch.save(layer_weights, file_path)

    # Define additional fields
    username = "example_user"
    iteration_number = 1

    # Create a dictionary with your fields
    data = {
        "username": username,
        "model_class_name": model.__class__.__name__,
        "iteration_number": iteration_number,
    }
    files = {"file": open(file_path, "rb")}
    payload = {"data": json.dumps(data)}

    request_response(endpoint, requests.post, payload, files)

    os.remove(file_path)


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    model = SmallCNN()

    init("test project", SmallCNN)

    example_input = torch.randn(1, 3, 255, 255)
    output = model(example_input)
    # file_path = "temp.pt"
    # torch.jit.save(torch.jit.script(model), file_path)

    send_model(model)


if __name__ == "__main__":
    run()
