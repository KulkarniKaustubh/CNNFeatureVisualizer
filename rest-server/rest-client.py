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
REST = os.getenv("REST") or "localhost:80"


# How to fix the problem when we have arguments in the init function
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

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
    file_path = "temp.pt"
    torch.jit.save(torch.jit.script(model), file_path)
    # Define additional fields
    username = "example_user"
    model_name = "example_model"
    iteration_number = 1

    # Create a dictionary with your fields
    data = {
        'username': username,
        'modelname': model_name,
        'iterationNumber': iteration_number,
    }
    files = {'file': open(file_path, 'rb')}
    payload = {'data': json.dumps(data)}
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
    response = reqmethod(f"http://{REST}/{endpoint}", files=files, data = payload)
    os.remove(file_path)
    if response.status_code == 200:
        jsonResponse = json.dumps(response.json(), indent=4, sort_keys=True)
        print(jsonResponse)
        return
    else:
        print(
            f"response code is {response.status_code}, raw response is {response.text}")
        return response.text

def sendmodel2(reqmethod, endpoint, debug=True):
    model = get_model()
    file_path = 'layer_weights.pth'
    layer_weights = {}
    for name, param in model.named_parameters():
        layer_weights[name] = param.clone().detach().cpu()

# Save the dictionary containing layer weights
    torch.save(layer_weights, file_path)
    # Define additional fields
    username = "example_user"
    model_name = "example_model"
    iteration_number = 1

    source_code_file_path='temp.py'
    with open(source_code_file_path, 'w') as f:
        f.write(inspect.getsource(SimpleModel))
    with open(source_code_file_path, 'r') as f:
        model_source_code = f.read()
    # Create a dictionary with your fields
    data = {
        'username': username,
        'modelname': model_name,
        'iterationNumber': iteration_number,
        'source_code' : model_source_code

    }

    # weights_file_path='temp_weights.pth'
    # 
    # torch.save(model, weights_file_path)
    # with open(source_code_file_path, 'r') as f:
    #     model_source_code = f.read()
    # state_dict_buffer = io.BytesIO()
    # torch.save(model.state_dict(), state_dict_buffer)

    # # files = {'weights': open(weights_file_path, 'rb')}
    # files = {'weights': state_dict_buffer}
    # data = {'source_code': model_source_code}

    
    files = {'file': open(file_path, 'rb')}
    payload = {'data': json.dumps(data)}
    
    # # files = {'weights': open(weights_file_path, 'rb')}
    # files = {'weights': state_dict_buffer}
    # data = {'source_code': model_source_code}
    response = reqmethod(f"http://{REST}/{endpoint}", files=files, data = payload)
    os.remove(file_path)
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
    sendmodel2(requests.post,'visualize2')
if __name__ == "__main__":
    run()
