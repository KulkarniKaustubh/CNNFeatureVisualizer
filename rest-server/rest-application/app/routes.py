from app import app
import json

import os

# import sys
from flask import (
    render_template,
    flash,
    redirect,
    request,
    make_response,
    Response,
    jsonify,
    send_file,
)
from flask import g

# import io
import hashlib
import time
import random

# import base64
import torch

# import psycopg2
# from psycopg2 import sql
import importlib
import torch.nn as nn  # Include the necessary import
import tempfile

import redis
from minio import Minio

conn = None
redisHost = os.getenv("REDIS_HOST") or "localhost"
redisPort = os.getenv("REDIS_PORT") or 6379
minioHost = os.getenv("MINIO_HOST") or "localhost"
minioPort = os.getenv("MINIO_PORT") or 9000
redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
minioUser = "rootuser"
minioPasswd = "rootpass123"
minioFinalAddress = minioHost + ":" + minioPort
minioClient = Minio(
    minioFinalAddress,
    secure=False,
    access_key=minioUser,
    secret_key=minioPasswd,
)
# bucketName = "queue"


def get_model_output(model):
    model.eval()
    example_input = torch.randn(1, 3, 255, 255)
    with torch.no_grad():
        output = model(example_input)
    return output


@app.route("/", methods=["GET"])
def hello():
    return "Hi. Welcome to the rest-server"


@app.route("/postgres/connect", methods=["GET"])
def testPostgresConnection():
    global conn
    try:
        # Connection object for the case it is deployed on the container
        # conn = psycopg2.connect(
        #         host='postgres',
        #         port=5432,
        #         user='admin',
        #         password='psltest',
        #         dbname='postgresdb'
        # )
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="admin",
            password="psltest",
            dbname="postgresdb",
        )
        return "Fuck yeah!"
    except Exception as e:
        return "Exception: " + e


@app.route("/postgres/createTable", methods=["POST"])
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
                    sql.SQL("{} {}").format(
                        sql.Identifier(column[0]), sql.SQL(column[1])
                    )
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
        response_data = {"response": "Failed", "Exception": e}
        return jsonify(response_data)


@app.route("/postgres/getTable", methods=["GET"])
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


@app.route("/postgres/getRows", methods=["GET"])
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
    response_data = {"response": "Success", "rows": rows}
    return jsonify(response_data)


def make_minio_bucket(bucket_name):
    if minioClient.bucket_exists(bucket_name):
        print(f"{bucket_name} Bucket exists")
    else:
        minioClient.make_bucket(bucket_name)
        print(f"Bucket {bucket_name} has been created")


def push_to_minio_bucket(
    bucket_name, minio_file_location, source_file_location
):
    result = minioClient.fput_object(
        bucket_name, minio_file_location, source_file_location
    )
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name,
            result.etag,
            result.version_id,
        )
    )


def _generate_hash() -> str:
    data = f"{time.time()}{random.random()}"

    # Generate SHA-1 hash
    sha1_hash = hashlib.sha1(data.encode()).hexdigest()

    # Truncate to 12 characters
    truncated_hash = sha1_hash[:12]

    return truncated_hash


@app.route("/initialize", methods=["POST"])
def init():
    data = json.loads(request.form.get("data"))

    project_name = data["project"].strip()
    project_name = project_name.replace(" ", "-")
    project_name += f"-{_generate_hash()}"

    source_code = data["model_source_code"]
    source_code_file_location = "received_model_source_code.py"
    with open(source_code_file_location, "w") as f:
        f.write(source_code)

    model_class_name = data["model_class_name"]
    bucket_name = "source-code"
    minio_file_location = model_class_name

    make_minio_bucket(bucket_name=bucket_name)
    push_to_minio_bucket(
        bucket_name=bucket_name,
        minio_file_location=minio_file_location,
        source_file_location=source_code_file_location,
    )
    response_data = {"response": "Success"}

    return jsonify(response_data)


@app.route("/visualize2", methods=["POST"])
def upload_model_2():
    uploaded_file = request.files["file"]
    layer_weights_file_location = "layer_weights.pth"
    uploaded_file.save(layer_weights_file_location)
    data = json.loads(request.form.get("data"))
    model_class_name = data["model_class_name"]
    iteration_number = data["iteration_number"]
    bucket_name = "layer-weights"
    minio_file_location = model_class_name + "/" + str(iteration_number)
    make_minio_bucket(bucket_name=bucket_name)
    push_to_minio_bucket(
        bucket_name=bucket_name,
        minio_file_location=minio_file_location,
        source_file_location=layer_weights_file_location,
    )
    redisClient.lpush(
        "toWorkers",
        f"Request for : {model_class_name}. Layer_Weights placed at: {minio_file_location}",
    )
    print("Pushed to redis queue")
    response_data = {"response": "Success"}
    return jsonify(response_data)
