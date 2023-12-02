import sys
import os
import redis
from minio import Minio
import torch
import torch.nn as nn
import importlib
redisHost = os.getenv("REDIS_HOST") or "localhost"
redisPort = os.getenv("REDIS_PORT") or 6379
minioHost = os.getenv("MINIO_HOST") or "localhost"
minioPort = os.getenv("MINIO_PORT") or 9000
redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
minioUser = "rootuser"
minioPasswd = "rootpass123"
minioFinalAddress = minioHost + ":" + minioPort
minioClient = Minio(minioFinalAddress,
               secure=False,
               access_key=minioUser,
               secret_key=minioPasswd)
def get_model_output(model):
    example_input = torch.randn(1, 3, 255, 255)
    output = model(example_input)
    return output

def get_model(source_code, layer_weights_file):
    spec = importlib.util.spec_from_file_location(
        "received_model_module", source_code
    )
    received_model_module = importlib.util.module_from_spec(spec)
    received_model_module.__dict__["torch"] = torch
    received_model_module.__dict__["nn"] = nn
    spec.loader.exec_module(received_model_module)

    # Get the actual class name dynamically
    model_class_name = [
        name
        for name, obj in received_model_module.__dict__.items()
        if isinstance(obj, type)
    ][0]

    # Use the actual class name to instantiate the model
    ReceivedModel = getattr(received_model_module, model_class_name)

    # Create an instance of the received model class
    received_model_instance = ReceivedModel()
    layer_weights = torch.load(layer_weights_file)

    print("received model is: ", received_model_instance)

    # Load each layer's weights back into the model
    for name, param in received_model_instance.named_parameters():
        param.data.copy_(layer_weights[name])
    # received_model_instance.to("mps")
    return received_model_instance
    
bucket_name_source_code = "source-code"
bucket_name_layer_weights = "layer-weights"
source_code_file_location = "received_model_source_code.py"
layer_weights_file_path = "layer_weights.pth"
while True:
    try:
        work = redisClient.blpop("toWorkers", timeout=0)
        model_name = work[1].decode('utf-8').split('.')[0].split(':')[1].strip()
        layer_weights_minio_path = work[1].decode('utf-8').split('.')[1].split(':')[1].strip()
        response=None
        # Get data of an object.
        try:
            print("Model Name : ", model_name)
            print("Layer_Weights Minio Path: ", layer_weights_minio_path)
            response = minioClient.fget_object(bucket_name_source_code, model_name, source_code_file_location)
            print("Recieved source code in location: ", source_code_file_location)
            response = minioClient.fget_object(bucket_name_layer_weights, layer_weights_minio_path, layer_weights_file_path)
            print("Recieved Layer Weights in location: ", layer_weights_file_path)
            model = get_model(source_code=source_code_file_location, layer_weights_file=layer_weights_file_path)
            print("Model has been Loaded Succesfully")
            output = get_model_output(model=model)
            print("Output is: ", output)       
    # Read data from response.
        finally:
            if response != None:
                response.close()
                response.release_conn()
    except Exception as exp:
        print(f"Exception raised in log loop: {str(exp)}")
    sys.stdout.flush()
    sys.stderr.flush()