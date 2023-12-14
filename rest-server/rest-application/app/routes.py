from app import app

import json
import os

from flask import request, jsonify, send_file

import psycopg2
from psycopg2 import sql

import redis
from minio import Minio

conn = None
redisHost = os.getenv("REDIS_HOST") or "localhost"
redisPort = os.getenv("REDIS_PORT") or 6379
minioHost = os.getenv("MINIO_HOST") or "localhost"
minioPort = os.getenv("MINIO_PORT") or "9000"
postgresHost = os.getenv("POSTGRES_HOST") or "localhost"
postgresPort = os.getenv("POSTGRES_PORT") or 5432
postgresUser = "admin"
postgresPassword = "psltest"
postgresDbname = "postgresdb"
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
# def get_model_output(model):
#     model.eval()
#     example_input = torch.randn(1, 3, 255, 255)
#     with torch.no_grad():
#         output = model(example_input)
#     return output
# def get_model_output(model):
#     model.eval()
#     example_input = torch.randn(1, 3, 255, 255)
#     with torch.no_grad():
#         output = model(example_input)
#     return output


@app.route("/", methods=["GET"])
def hello():
    return "Hi. Welcome to the rest-server"


# Move this to the Init of the Rest-Server. We should always connect to Postgres first before doing all the operations
# Function to establish a PostgreSQL connection
def connect_to_postgres():
    global conn
    try:
        conn = psycopg2.connect(
            host=postgresHost,
            port=postgresPort,
            user=postgresUser,
            password=postgresPassword,
            dbname=postgresDbname,
        )
        print("PostgreSQL connection established.")
    except Exception as e:
        print("Exception: " + str(e))


# Execute the connect_to_postgres function before the first request
@app.before_request
def before_request():
    if not hasattr(app, "postgres_connected"):
        connect_to_postgres()
        app.postgres_connected = True


def create_table_helper(table_name, columns):
    try:
        with conn.cursor() as cursor:
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

        response_data = {"response": "Success. Table has been created"}
        return jsonify(response_data)

    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
        return jsonify(response_data)


@app.route("/postgres/createTable", methods=["POST"])
def create_custom_table():
    try:
        # Get data from the request body
        request_data = request.get_json()
        table_name = request_data.get("table_name")
        columns = request_data.get("columns")

        # Check if required data is present
        if not table_name or not columns:
            raise ValueError(
                "Table name and columns are required in the request body"
            )

        # Call the create_table function with the provided parameters
        response = create_table_helper(table_name, columns)
        return response

    except Exception as e:
        response_data = {"response": "Failed", "Exception": str(e)}
        return jsonify(response_data)


@app.route("/postgres/getSchema", methods=["POST"])
def get_schema():
    try:
        # Get data from the request body
        request_data = request.get_json()
        table_name = request_data.get("table_name")

        # Check if required data is present
        if not table_name:
            raise ValueError("Table name is required in the request body")

        with conn.cursor() as cursor:
            query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'  -- You can adjust this condition based on your schema
            AND table_name = %s;
            """

            # Execute the query with the table name parameter
            cursor.execute(query, (table_name,))

            # Fetch all the results
            columns_info = cursor.fetchall()

            # Display the result
            for column_info in columns_info:
                print(
                    f"Column Name: {column_info[0]}, Data Type: {column_info[1]}"
                )

        cursor.close()
        response_data = {"response": "Success"}
        return jsonify(response_data)

    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
        return jsonify(response_data)


@app.route("/postgres/deleteTable", methods=["POST"])
def delete_table():
    try:
        # Get data from the request body
        request_data = request.get_json()
        table_name = request_data.get("table_name")

        # Check if required data is present
        if not table_name:
            raise ValueError("Table name is required in the request body")

        with conn.cursor() as cursor:
            # Generate the DROP TABLE statement
            drop_table_query = sql.SQL(
                "DROP TABLE IF EXISTS {} CASCADE;"
            ).format(sql.Identifier(table_name))

            # Execute the DROP TABLE statement
            cursor.execute(drop_table_query)
            conn.commit()

        cursor.close()
        response_data = {
            "response": f"Success. Table '{table_name}' has been deleted"
        }
        return jsonify(response_data)

    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
        return jsonify(response_data)


@app.route("/postgres/getRows", methods=["POST"])
def get_rows():
    try:
        # Get data from the request body
        request_data = request.get_json()
        table_name = request_data.get("table_name")

        # Check if required data is present
        if not table_name:
            raise ValueError("Table name is required in the request body")

        with conn.cursor() as cursor:
            # Generate the SELECT query with the provided table name
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

    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
        return jsonify(response_data)


@app.route("/postgres/insertRows", methods=["POST"])
def insert_rows():
    try:
        # Get data from the request body
        request_data = request.get_json()
        table_name = request_data.get("table_name")
        rows_to_insert = request_data.get("rows")

        # Check if required data is present
        if not table_name or not rows_to_insert:
            raise ValueError(
                "Table name and rows are required in the request body"
            )

        with conn.cursor() as cursor:
            # Generate the INSERT query with the provided table name and rows
            insert_query = sql.SQL("INSERT INTO {} VALUES {};").format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(
                    sql.SQL("({})").format(
                        sql.SQL(", ").join(sql.Literal(value) for value in row)
                    )
                    for row in rows_to_insert
                ),
            )

            # Execute the INSERT query
            cursor.execute(insert_query)
            conn.commit()

        cursor.close()
        response_data = {"response": "Success. Rows have been inserted"}
        return jsonify(response_data)

    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
        return jsonify(response_data)


@app.route("/postgres/getLosses", methods=["POST"])
def get_losses_epochs():
    try:
        # Get data from the request body
        request_data = request.get_json()
        model_hash = request_data.get("model_hash")
        username = request_data.get("username")

        # Check if required data is present
        if not model_hash or not username:
            raise ValueError(
                "Both model_hash and username are required in the request body"
            )

        with conn.cursor() as cursor:
            # Define the query to select losses and epochs for a specific model_hash and username
            query = sql.SQL(
                "SELECT epoch, losses FROM training_metrics WHERE model_hash = {} AND username = {};"
            ).format(sql.Literal(model_hash), sql.Literal(username))

            # Execute the query
            cursor.execute(query)

            # Fetch all rows
            rows = cursor.fetchall()

        cursor.close()
        # Format the result as a list of dictionaries
        epochs = []
        losses = []
        result = [{"epoch": row[0], "losses": row[1]} for row in rows]
        epochs = [row[0] for row in rows]
        losses = [row[1] for row in rows]
        print("Epochs : ", epochs)
        print("Losses : ", losses)
        print("-----------------------------------------------")
        response_data = {"response": "Success", "data": result}
        return jsonify(response_data)

    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
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


@app.route("/initialize", methods=["POST"])
def init():
    data = json.loads(request.form.get("data"))

    source_code = data["model_source_code"]
    source_code_file_location = "received_model_source_code.py"
    with open(source_code_file_location, "w") as f:
        f.write(source_code)

    # model_class_name = data["model_class_name"]
    project_id = data["project_id"]
    bucket_name = "source-code"
    minio_file_location = project_id

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
    # model_class_name = data["model_class_name"]
    user_name = data["username"]
    project_id = data["project_id"]
    iteration_number = data["iteration_number"]
    bucket_name = "layer-weights"
    minio_file_location = project_id + "/" + str(iteration_number)
    make_minio_bucket(bucket_name=bucket_name)
    push_to_minio_bucket(
        bucket_name=bucket_name,
        minio_file_location=minio_file_location,
        source_file_location=layer_weights_file_location,
    )
    redisClient.lpush(
        "toWorkers",
        f"Request by : {user_name}. Request for : {project_id}. Layer_Weights placed at: {minio_file_location}",
    )
    print("Pushed to redis queue")
    response_data = {"response": "Success"}
    return jsonify(response_data)

@app.route("/downloadVis", methods=["POST"])
def downloadVis():
    data = json.loads(request.form.get("data"))
    project_id = data["project_id"]
    minio_file_location = f"{project_id}.zip"
    bucket_name = "visualizations"
    zip_file_location = "generated_vis.zip"
    response = minioClient.fget_object(
            bucket_name, minio_file_location, zip_file_location
    )
    print("Recieved Layer Weights in location: ", minio_file_location)
    return send_file(zip_file_location, as_attachment=True, download_name='generated_visualizations.zip')