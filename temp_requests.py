import requests

def create_table(table_name, columns):
    url = 'http://34.121.134.170.nip.io:80/postgres/createTable'
    headers = {'Content-Type': 'application/json'}

    data = {
        "table_name": table_name,
        "columns": columns
    }
    response = requests.post(url, json=data, headers=headers)
    return response.status_code, response.json()

def insert_rows(table_name, rows):
    url = 'http://34.121.134.170.nip.io:80/postgres/insertRows'
    headers = {'Content-Type': 'application/json'}
    data = {
        "table_name": table_name,
        "rows": rows,
    }
    response = requests.post(url, json=data, headers=headers)
    return response.status_code, response.json()


def create_user_table():
    table_name = "users"
    columns = [
        ["username", "VARCHAR PRIMARY KEY"]
    ]
    status_code, response_data = create_table(table_name, columns)
    print(f"Status Code: {status_code}")
    print("Response:", response_data)

def create_model_hashes_table():
    table_name = "model_hashes"
    columns = [
        ["model_hash", "VARCHAR PRIMARY KEY"],
        ["username", "VARCHAR"]
    ]

    status_code, response_data = create_table(table_name, columns)

    print(f"Status Code: {status_code}")
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
    status_code, response_data = create_table(table_name, columns)

    print(f"Status Code: {status_code}")
    print("Response:", response_data)

    status_code, response_data = create_table(table_name, columns)

    print(f"Status Code: {status_code}")
    print("Response:", response_data)

def insert_to_users(username):
    table_name = "users"
    rows = [
        [username]
    ]
    status_code, response_data = insert_rows(table_name, rows)
    print(f"Status Code: {status_code}")
    print("Response:", response_data)

def insert_to_model_hashes():
    table_name = "model_hashes"
    rows = [
        ["67890^&*()", "john_doe"]
    ]
    status_code, response_data = insert_rows(table_name, rows)
    print(f"Status Code: {status_code}")
    print("Response:", response_data)

def insert_to_training_metrics():
    table_name = "training_metrics"
    rows = [
        ["jayant_duneja", "123456", 2,None,0.5, None, 0.7, None]
    ]
    status_code, response_data = insert_rows(table_name, rows)
    print(f"Status Code: {status_code}")
    print("Response:", response_data)

def get_losses():
    url = 'http://34.121.134.170.nip.io:80/postgres/getLosses'
    headers = {'Content-Type': 'application/json'}

    data = {
        "username": "john_doe",
        "model_hash": "67890^&*()"
    }

    response = requests.post(url, json=data, headers=headers)

    print(response.status_code)
    print(response.json())


# create_user_table()
# create_model_hashes_table()
# create_training_metrics_table()
# insert_to_users()
# insert_to_model_hashes()
insert_to_training_metrics()
# get_losses()