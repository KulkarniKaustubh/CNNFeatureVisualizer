from typing import Tuple
import subprocess
import redis
from minio import Minio
import time
import torchboard as tb
import os
import glob
import io

import torch
import torch.nn as nn

from torchboard.utils import load_model_weights


def _get_model(source_code, layer_weights_file):
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

    received_model_instance = load_model_weights(
        received_model_instance, layer_weights_file
    )

    return received_model_instance


def init_redis() -> redis.StrictRedis:
    redis_host = os.getenv("REDIS_HOST") or "localhost"
    redis_port = os.getenv("REDIS_PORT") or 6379

    redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    return redis_client


def init_minio() -> Minio:
    minio_host = os.getenv("MINIO_HOST") or "localhost"
    minio_port = os.getenv("MINIO_PORT") or 9000
    minio_user = os.getenv("MINIO_USER") or "rootuser"
    minio_passwd = os.getenv("MINIO_PASSWD") or "rootpass123"

    minio_final_addr = f"{minio_host}:{minio_port}"

    minio_client = Minio(
        minio_final_addr,
        secure=False,
        access_key=minio_user,
        secret_key=minio_passwd,
    )

    return minio_client


def check_minio_objects(minio_client, bucket_name):
    minio_objects = minio_client.list_objects(bucket_name)

    if minio_objects is not None:
        print(f"Objects: {minio_objects}")
        for obj in minio_objects:
            print(
                obj.bucket_name,
                obj.object_name,
                obj.last_modified,
                obj.etag,
                obj.size,
                obj.content_type,
            )
    else:
        print("Minio objects are empty or do not exist.")


def get_model_from_queue(
    redis_client: redis.StrictRedis, minio_client: Minio
) -> nn.Module:
    redis_queue = "toWorkers"

    bucket_name_source_code = "source-code"
    bucket_name_layer_weights = "layer-weights"
    source_code_file_location = "received_model_source_code.py"
    layer_weights_file_path = "layer_weights.pth"

    work = redis_client.blpop(redis_queue, timeout=0)

    model_name = work[1].decode("utf-8").split(".")[0].split(":")[1].strip()
    layer_weights_minio_path = (
        work[1].decode("utf-8").split(".")[1].split(":")[1].strip()
    )

    model = None
    response = None

    try:
        print("Model Name : ", model_name)
        print("Layer_Weights Minio Path: ", layer_weights_minio_path)
        response = minio_client.fget_object(
            bucket_name_source_code, model_name, source_code_file_location
        )
        print("Recieved source code in location: ", source_code_file_location)
        response = minio_client.fget_object(
            bucket_name_layer_weights,
            layer_weights_minio_path,
            layer_weights_file_path,
        )
        print("Recieved Layer Weights in location: ", layer_weights_file_path)

        model = _get_model(
            source_code=source_code_file_location,
            layer_weights_file=layer_weights_file_path,
        )
    finally:
        if response is not None:
            response.close()
            response.release_conn()

    return model


def main():
    redis_client = init_redis()
    minio_client = init_minio()

    while True:
        model = get_model_from_queue(redis_client, minio_client)

        cnn_layer_viz = tb.visualizers.CNNLayerVisualization(model, 0, 5)

        # print("Visualizing with hooks.")
        # cnn_layer_viz.visualise_layer_with_hooks()

        print("Visualizing without hooks.")
        cnn_layer_viz.visualise_layer_without_hooks()


if __name__ == "__main__":
    try:
        main()
    except Exception as exp:
        print(f"Exception raised in main(): {str(exp)}")

    # bucket_name = "queue"
    # redis_queue = "toWorker"
    # output_bucket_name = "output"
    #
    # if not minio_client.bucket_exists(bucket_name):
    #     print(f"Create bucket {bucket_name}")
    #     minio_client.make_bucket(bucket_name)
    #
    # if not minio_client.bucket_exists(output_bucket_name):
    #     print(f"Create bucket {output_bucket_name}")
    #     minio_client.make_bucket(output_bucket_name)

    # while True:
    #     print("In while")
    #     try:
    #         print("Checking minio queue.")
    #         check_minio_objects(minio_client, bucket_name)
    #
    #         print("Checking minio bucket output.")
    #         check_minio_objects(minio_client, output_bucket_name)
    #
    #         print("Checking redis queue.")
    #         work = redis_client.blpop(redis_queue, timeout=0)
    #         print(work)
    #
    #         song_hash = work[1].decode("utf-8")
    #         local_name = f"/data/input/{song_hash}.mp3"
    #         print(work[1].decode("utf-8"))
    #
    #         print("Obtain song object from minio.")
    #         resp = minio_client.fget_object(
    #             "queue", work[1].decode("utf-8"), local_name
    #         )
    #
    #         print("Running demucs command.")
    #         subprocess.run(
    #             [
    #                 "python3",
    #                 "-m",
    #                 "demucs.separate",
    #                 "--mp3",
    #                 "--out",
    #                 "/data/output",
    #                 local_name,
    #             ]
    #         )
    #
    #         print("Uploading song")
    #         for mp3 in glob.glob(f"/data/output/htdemucs/{song_hash}/*mp3"):
    #             track_name = mp3.split("/")[-1]  # getting only the filename
    #             obj_name = f"{song_hash}/{track_name}"
    #             mp3_stream = io.BytesIO(open(mp3, "rb").read())
    #
    #             minio_client.put_object(
    #                 output_bucket_name,
    #                 obj_name,
    #                 mp3_stream,
    #                 mp3_stream.getbuffer().nbytes,
    #             )
    #             print(f"Uploaded object: {obj_name}")
    #
    #     except Exception as exp:
    #         traceback.print_exc()
    #         print(f"Exception raised in log loop: {str(exp)}")

    # while True:
    #     print("Hello from the worker!")
    #     time.sleep(2)
