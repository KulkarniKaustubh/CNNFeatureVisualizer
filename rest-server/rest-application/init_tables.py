import psycopg2
from psycopg2 import sql
import os
print("Hellooooooooooo")
conn  = None
postgresHost = os.getenv("POSTGRES_HOST") or "localhost"
postgresPort = os.getenv("POSTGRES_PORT") or 5432
postgresUser = 'admin'
postgresPassword = 'psltest'
postgresDbname = 'postgresdb'
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

# # Execute the connect_to_postgres function before the first request
# def before_request():
#     if not hasattr(app, 'postgres_connected'):
#         connect_to_postgres()
#         app.postgres_connected = True


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
        return response_data
    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
        return response_data

def create_custom_table(request_data):
    try:
        # Get data from the request body
        # request_data = request.get_json()
        table_name = request_data.get("table_name")
        columns = request_data.get("columns")

        # Check if required data is present
        if not table_name or not columns:
            raise ValueError("Table name and columns are required in the request body")

        # Call the create_table function with the provided parameters
        response = create_table_helper(table_name, columns)
        return response

    except Exception as e:
        response_data = {"response": "Failed", "Exception": str(e)}
        return response_data

def get_schema(request):
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
                print(f"Column Name: {column_info[0]}, Data Type: {column_info[1]}")

        cursor.close()
        response_data = {"response": "Success"}
        return response_data

    except Exception as e:
        conn.rollback()
        response_data = {"response": "Failed", "Exception": str(e)}
        return response_data

def create_table(table_name, columns):
    
    data = {
        "table_name": table_name,
        "columns": columns
    }
    response = create_custom_table(data)
    return response

def create_user_table():
    table_name = "users"
    columns = [
        ["username", "VARCHAR PRIMARY KEY"]
    ]
    response_data = create_table(table_name, columns)
    print("Response:", response_data)


def create_model_hashes_table():
    table_name = "model_hashes"
    columns = [
        ["model_hash", "VARCHAR PRIMARY KEY"],
        ["username", "VARCHAR"]
    ]

    response_data = create_table(table_name, columns)

    print("Response:", response_data)

def create_training_metrics_table():
    table_name = "training_metrics"
    columns = [
        ["epoch", "INT"],
        ["losses", "FLOAT"],
        ["accuracy", "FLOAT"],
        ["model_hash", "VARCHAR"],
        ["username", "VARCHAR"]
    ]
    response_data = create_table(table_name, columns)

    print("Response:", response_data)

    
if name == '__main__':
   create_user_table()
   create_model_hashes_table()
   create_training_metrics_table()