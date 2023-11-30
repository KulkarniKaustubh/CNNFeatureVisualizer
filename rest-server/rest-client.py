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
REST = os.getenv("REST") or "localhost:5000"


# How to fix the problem when we have arguments in the init function
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

def get_model():
    # Create an instance of the model
    input_size = 10  # Adjust this based on your input size
    hidden_size = 20  # Adjust as needed
    output_size = 1  # Binary classification, so output size is 1
    model = SimpleModel(input_size, hidden_size, output_size)
    print("Model is: ")
    print(model)
    output = get_model_output(model=model)
    print("Model output: ")
    print(output)
    return model

def get_model_output(model):
    model.eval()
    example_input = torch.ones((1, 10))
    with torch.no_grad():
        output = model(example_input)
    return output.item()

def sendmodel(reqmethod, endpoint, debug=True):
    model = get_model()
    source_code_file_path='temp.py'
    weights_file_path='temp_weights.pth'
#     torch.jit.save(torch.jit.script(model), file_path)
# # Read the contents of the file as binary
#     with open(file_path, "rb") as file:
#         model_bytes = file.read()
#     data = {
#         'model_data' : base64.b64encode(model_bytes).decode('utf-8')
#     }
#     jsonData = jsonpickle.encode(data)
    with open(source_code_file_path, 'w') as f:
        f.write(inspect.getsource(SimpleModel))
    # Save the model's weights
    # torch.save(model.state_dict(), weights_file_path)
    torch.save(model, weights_file_path)
    with open(source_code_file_path, 'r') as f:
        model_source_code = f.read()
    files = {'weights': open(weights_file_path, 'rb')}
    data = {'source_code': model_source_code}
    # response = reqmethod(f"http://{REST}/{endpoint}", data=jsonData, headers={'Content-type': 'application/json'})
    response = reqmethod(f"http://{REST}/{endpoint}", files=files, data=data)
    if response.status_code == 200:
        jsonResponse = json.dumps(response.json(), indent=4, sort_keys=True)
        print(jsonResponse)
        return
    else:
        print(
            f"response code is {response.status_code}, raw response is {response.text}")
        return response.text

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    sendmodel(requests.post,'upload')
if __name__ == "__main__":
    run()
