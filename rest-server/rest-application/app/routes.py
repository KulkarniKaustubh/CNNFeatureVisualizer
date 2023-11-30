from app import app
import json
import os
import sys
from flask import render_template, flash, redirect, request, make_response, Response, jsonify, send_file
import io
import hashlib
import base64
import torch
import psycopg2
import importlib
import torch.nn as nn  # Include the necessary import
from received_model_source_code import *
def get_model_output(model):
    model.eval()
    example_input = torch.ones((1, 10))
    with torch.no_grad():
        output = model(example_input)
    return output.item()
@app.route('/', methods=['GET'])
def hello():
    return 'Hi. Welcome to the rest-server'
@app.route('/postgres/connect', methods=['GET'])
def testPostgresConnection():
        try:
                conn = psycopg2.connect(
                        host='postgres',
                        port=5432,
                        user='admin',
                        password='psltest',
                        dbname='postgresdb'
                )
                return "Fuck yeah!"
        except Exception as e:
              return "Exception: " + e
@app.route('/upload', methods=['POST'])
def upload():
        # data = request.get_json()
        # # print(data)
        # model_data = data.get('model_data')
        # # print(model_data)
        # model_data_decoded = base64.b64decode(model_data)
        # print(type(model_data_decoded))
        # model_data_io = io.BytesIO(model_data_decoded)
        # # model_data.seek(0)
        # model = torch.jit.load(model_data_io)
        # print("Printing the Model: ")
        # print(model)
        # output = get_model_output(model=model)
        # print("Model output: ")
        # print(output)
        # response_data = {"response" : "Success"}
        # return jsonify(response_data)
        source_code = request.form['source_code']
        weights_file = request.files['weights']

        # Save the source code to a file (optional)
        with open('received_model_source_code.py', 'w') as f:
            f.write(source_code)

        # Import the model class dynamically
        # spec = importlib.util.spec_from_loader("ReceivedModel", loader=None)
        # received_model = importlib.util.module_from_spec(spec)
        # received_model.__dict__['torch'] = torch
        # received_model.__dict__['nn'] = nn
        # exec(source_code, received_model.__dict__)
        #  # Get the actual class name dynamically
        # model_class_name = [name for name, obj in received_model.__dict__.items() if isinstance(obj, type)][0]

        # # Use the actual class name to instantiate the model
        # ReceivedModel = getattr(received_model, model_class_name)
        # # ReceivedModel = received_model.ReceivedModel

        # # Create an instance of the received model class
        # received_model_instance = ReceivedModel()

        # # Load the received model's weights
        # received_model_instance.load_state_dict(torch.load(weights_file))
        received_model_instance = torch.load(weights_file)
        output = get_model_output(model=received_model_instance)
        print("Model output: ")
        print(output)

        # Perform any operations with the received model instance

        response_data = {"response" : "Success"}
        return jsonify(response_data)


