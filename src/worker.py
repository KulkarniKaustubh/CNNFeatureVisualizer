import subprocess
import redis
from minio import Minio
import time
import torchboard as tb
import os
import glob
import io


def init_redis() -> redis.StrictRedis:
    redis_host = os.getenv("REDIS_HOST") or "localhost"
    redis_port = os.getenv("REDIS_PORT") or 6379

    redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    return redis_client


def init_minio() -> Minio:
    minio_host = os.getenv("MINIO_HOST") or "minio:9000"
    minio_user = os.getenv("MINIO_USER") or "rootuser"
    minio_passwd = os.getenv("MINIO_PASSWD") or "rootpass123"

    minio_client = Minio(
        minio_host,
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


def get_model_from_queue():
    model = None

    cnn_layer_viz = tb.visualizers.CNNLayerVisualization(model, 0, 5)

    # print("Visualizing with hooks.")
    # cnn_layer_viz.visualise_layer_with_hooks()

    print("Visualizing without hooks.")
    cnn_layer_viz.visualise_layer_without_hooks()


if __name__ == "__main__":
    redis_client = init_redis()
    minio_client = init_minio()
    print(minio_client)

    bucket_name = "queue"
    redis_queue = "toWorker"
    output_bucket_name = "output"

    if not minio_client.bucket_exists(bucket_name):
        print(f"Create bucket {bucket_name}")
        minio_client.make_bucket(bucket_name)

    if not minio_client.bucket_exists(output_bucket_name):
        print(f"Create bucket {output_bucket_name}")
        minio_client.make_bucket(output_bucket_name)

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

    while True:
        print("Hello from the worker!")
        time.sleep(2)
