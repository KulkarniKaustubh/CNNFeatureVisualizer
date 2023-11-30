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
from psycopg2 import sql
import importlib
import torch.nn as nn  # Include the necessary import
import tempfile
# from received_model_source_code import *
conn = None
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
        global conn
        try:
                # conn = psycopg2.connect(
                #         host='postgres',
                #         port=5432,
                #         user='admin',
                #         password='psltest',
                #         dbname='postgresdb'
                # )
                conn = psycopg2.connect(
                        host='localhost',
                        port=5432,
                        user='admin',
                        password='psltest',
                        dbname='postgresdb'
                )
                return "Fuck yeah!"
        except Exception as e:
              return "Exception: " + e
@app.route('/postgres/createTable', methods = ['POST'])
def create_table():
    try:
        with conn.cursor() as cursor:
            # Define the table name
            table_name = "modeloutput"

            # Define the column names and data types
            columns = [
                ("username", "VARCHAR PRIMARY KEY"),
                ("modelname", "VARCHAR"),
                ("iterationnumber", "INT"),
                ("output", "FLOAT"),
            ]

            # Generate the CREATE TABLE statement
            create_table_query = sql.SQL("CREATE TABLE {} ({});").format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(
                    sql.SQL("{} {}").format(sql.Identifier(column[0]), sql.SQL(column[1]))
                    for column in columns
                ),
            )

            # Execute the CREATE TABLE statement
            cursor.execute(create_table_query)
            conn.commit()
        cursor.close()
        response_data = {"response": "Success. Table has been created"}
        return jsonify(response_data)
    except Exception as e:
        response_data = {"response": "Failed", "Exception" : e}
        return jsonify(response_data)



@app.route('/postgres/getTable', methods = ['GET'])
def get_table():
    with conn.cursor() as cursor:
        query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'  -- You can adjust this condition based on your schema
    AND table_type = 'BASE TABLE';
    """

    # Execute the query
        cursor.execute(query)

    # Fetch all the results
        table_names = cursor.fetchall()

    # Display the result
        for table_name in table_names:
            print(table_name[0])
    cursor.close()
    response_data = {"response": "Success"}
    return jsonify(response_data)

@app.route('/postgres/getRows', methods = ['GET'])
def get_rows():
    with conn.cursor() as cursor:
        table_name = "modeloutput"

        # Generate the SELECT query
        select_query = f"SELECT * FROM {table_name};"

        # Execute the SELECT query
        cursor.execute(select_query)

        # Fetch all rows
        rows = cursor.fetchall()
        for row in rows:
             print(row)
# Display the result
    cursor.close()
    response_data = {"response": "Success", 'rows' : rows}
    return jsonify(response_data)

@app.route('/visualize', methods=['POST'])
def upload_model():
    data = json.loads(request.form.get('data'))
    user_name = data['username']
    model_name = data['modelname']
    iteration_number = data['iterationNumber']
    uploaded_file = request.files['file']
    uploaded_file.save('uploaded_model.pt')
    loaded_model = torch.jit.load('uploaded_model.pt')
    output = get_model_output(model=loaded_model)
    # Adding the entry to the postgres table
    with conn.cursor() as cursor:
        # Define the table name
        table_name = "modeloutput"

        # Define the values for the new row
        values = (user_name, model_name, iteration_number, output)

        # Generate the INSERT INTO statement
        insert_query = f"INSERT INTO {table_name} (username, modelname, iterationnumber, output) VALUES (%s, %s, %s, %s);"

        # Execute the INSERT INTO statement
        cursor.execute(insert_query, values)
# Commit the changes
        conn.commit()
    response_data = {"response": "Success", "output": output}
    return jsonify(response_data)
# def instantiate_model(source_code, weights_file):
#     # # Save the source code to a file
#     with open('received_model_source_code.py', 'w') as f:
#         f.write(source_code)

#     # Import the model class dynamically
#     spec = importlib.util.spec_from_loader("ReceivedModel", loader=None)
#     received_model = importlib.util.module_from_spec(spec)
    
#     # Include necessary imports in the module's namespace
#     received_model.__dict__['torch'] = torch
#     received_model.__dict__['nn'] = nn
    
#     exec(source_code, received_model.__dict__)

#     # Get the actual class name dynamically
#     model_class_name = [name for name, obj in received_model.__dict__.items() if isinstance(obj, type)][0]

#     # Use the actual class name to instantiate the model
#     ReceivedModel = getattr(received_model, model_class_name)

#     # # Create an instance of the received model class
#     received_model_instance = ReceivedModel()

#     # Load the received model's weights
#     received_model_instance.load_state_dict(torch.load(weights_file))

#     return received_model_instance
# def load_state_dict_from_file(file):
#     # Save the file content to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(file.read())
#         temp_file_path = temp_file.name

#     # Load the state_dict from the temporary file
#     state_dict = torch.load(temp_file_path)

#     # Clean up the temporary file
#     if temp_file_path:
#         os.unlink(temp_file_path)

#     return state_dict


# @app.route('/upload', methods=['POST'])
# def upload_model():
#     # Get the source code and weights from the request
#     source_code = request.form['source_code']
#     weights_file = request.files['weights']
#     with open('received_model_source_code.py', 'w') as f:
#         f.write(source_code)

#     spec = importlib.util.spec_from_file_location("received_model_module", "received_model_source_code.py")
#     received_model_module = importlib.util.module_from_spec(spec)
#     received_model_module.__dict__['torch'] = torch
#     # received_model_module.__dict__['nn'] = nn
#     spec.loader.exec_module(received_model_module)

#     # Get the actual class name dynamically
#     model_class_name = [name for name, obj in received_model_module.__dict__.items() if isinstance(obj, type)][0]

#     # Use the actual class name to instantiate the model
#     ReceivedModel = getattr(received_model_module, model_class_name)

#     # Create an instance of the received model class
#     received_model_instance = ReceivedModel()
#     # received_model_module.__dict__['SimpleModel'] = ReceivedModel
#     # spec.loader.exec_module(received_model_module)
#     print(received_model_instance)
#     print(model_class_name)
#     # state_dict_buffer = io.BytesIO(weights_file.read())
#     with torch.no_grad():
#     # Load the received model's weights
#         # received_model_instance.load_state_dict(torch.load(weights_file))
#         received_model_instance.load_state_dict(load_state_dict_from_file(weights_file))

#     # Run the forward pass to get the model output
#     output = get_model_output(model=received_model_instance)

#     response_data = {"response": "Success", "output": output.tolist()}
#     return jsonify(response_data)
