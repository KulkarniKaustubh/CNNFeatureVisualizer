import redis
from minio import Minio
import torchboard as tb
import os
import importlib
import shutil

import torch
import torch.nn as nn


import torchboard.utils as tbu


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

    received_model_instance = tbu.load_model_weights(
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


def make_minio_bucket(minio_client: Minio, bucket_name: str):
    if minio_client.bucket_exists(bucket_name):
        print(f"{bucket_name} Bucket exists")
    else:
        minio_client.make_bucket(bucket_name)
        print(f"Bucket {bucket_name} has been created")


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

    model = None
    response = None

    try:
        work = redis_client.blpop(redis_queue, timeout=0)
        user_name = work[1].decode("utf-8").split(".")[0].split(":")[1].strip()
        # model_name = (
        #     work[1].decode("utf-8").split(".")[0].split(":")[1].strip()
        # )
        project_id = (
            work[1].decode("utf-8").split(".")[1].split(":")[1].strip()
        )
        layer_weights_minio_path = (
            work[1].decode("utf-8").split(".")[2].split(":")[1].strip()
        )

        print("Project_id : ", project_id)
        print("Layer_Weights Minio Path: ", layer_weights_minio_path)
        response = minio_client.fget_object(
            bucket_name_source_code, project_id, source_code_file_location
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
    except Exception as e:
        print("Exception: ", e)

    return model, user_name, project_id


def get_metric_dict_from_queue(
    redis_client: redis.StrictRedis, minio_client: Minio
) -> dict:
    return {}


def push_to_minio_bucket(
    minio_client, bucket_name, minio_file_location, source_file_location
):
    result = minio_client.fput_object(
        bucket_name, minio_file_location, source_file_location
    )
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name,
            result.etag,
            result.version_id,
        )
    )


def send_visualizations(
    minio_client: Minio, username: str, project_id: str
) -> None:
    print("Zipping...")
    tbu.zip_folder(f"{username}-{project_id}-generated")

    if os.path.exists(f"{username}-{project_id}-generated.zip"):
        print("Created zip file.")
    else:
        print("Did not create zip file.")

    bucket_name = "visualizations"
    file_location = f"{username}-{project_id}-generated.zip"

    make_minio_bucket(minio_client, bucket_name)

    push_to_minio_bucket(
        minio_client=minio_client,
        bucket_name=bucket_name,
        minio_file_location=file_location,
        source_file_location=file_location,
    )
    # minio_client.fput_object()

    return


def send_graphs(minio_client: Minio) -> None:
    return


def create_graphs(metric_dict: dict) -> None:
    graphs_dir = "/graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    if metric_dict["epoch"] is not None:
        metrics = [
            "train-loss",
            "train-acc",
            "val-loss",
            "val-acc",
            "test-loss",
            "test-acc",
        ]
        for metric in metrics:
            if metric_dict[metric] is not None:
                fig = tbu.plot_graph(
                    metric_dict["epoch"],
                    metric_dict[metric],
                    f"{metric} graph",
                    "epoch",
                    metric,
                )

                print(
                    f"Saving graph for {metric} in {graphs_dir}/{metric}-graph.png"
                )
                fig.savefig(f"{graphs_dir}/{metric}-graph.png")

                del fig
    return


def run_visualizations(
    model: nn.Module, username: str, project_id: str
) -> None:
    conv_layer_idx_dict = tbu.get_conv_layer_idx_dict(model)

    for idx in conv_layer_idx_dict:
        print(idx)
        for channel in range(conv_layer_idx_dict[idx]):
            print(channel)
            cnn_layer_viz = tb.visualizers.CNNLayerVisualization(
                model,
                idx,
                channel,
                visualizations_dir=f"{username}-{project_id}-generated",
            )

            print(f"Visualizing {idx} with hooks.")
            cnn_layer_viz.visualise_layer_with_hooks()

            print(f"Visualizing {idx} without hooks.")
            cnn_layer_viz.visualise_layer_without_hooks()

            # remove object from memory
            del cnn_layer_viz


def main():
    redis_client = init_redis()
    minio_client = init_minio()

    while True:
        model, username, project_id = get_model_from_queue(
            redis_client, minio_client
        )

        # metric_dict = get_metric_dict_from_queue(redis_client, minio_client)

        run_visualizations(model, username, project_id)
        # create_graphs(metric_dict)

        send_visualizations(minio_client, username, project_id)
        # send_graphs(minio_client)


if __name__ == "__main__":
    main()
