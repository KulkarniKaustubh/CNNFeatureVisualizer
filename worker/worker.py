import sys
import os
import redis
from minio import Minio
import torch
# redisHost = os.getenv("REDIS_HOST") or "localhost"
# redisPort = os.getenv("REDIS_PORT") or 6379
redisHost = "localhost"
redisPort = 6379
redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
# redisHost = "localhost"
# redisPort = 6379
# # minioHost = os.getenv("MINIO_HOST") or "localhost"
# # minioPort = os.getenv("MINIO_PORT") or 9000
minioUser = "rootuser"
minioPasswd = "rootpass123"
minioFinalAddress = 'localhost:9000'
minioClient = Minio(minioFinalAddress,
               secure=False,
               access_key=minioUser,
               secret_key=minioPasswd)
# redisClient = redis.StrictRedis(host=redisHost, port=redisPort, db=0)
bucketName = "output"
def get_model_output(model):
    model.eval()
    example_input = torch.ones((1, 10))
    with torch.no_grad():
        output = model(example_input)
    return output.item()
while True:
    try:
        work = redisClient.blpop("toWorkers", timeout=0)
        ##
        ## Work will be a tuple. work[0] is the name of the key from which the data is retrieved
        ## and work[1] will be the text log message. The message content is in raw bytes format
        ## e.g. b'foo' and the decoding it into UTF-* makes it print in a nice manner.
        ##
        file_location = work[1].decode('utf-8').split(':')[1].strip()
        bucketName = "queue"
        response=None
        # Get data of an object.
        try:
            response = minioClient.fget_object(bucketName, file_location, './uploaded_model.pt')
            print("Recieved Object in Location: ", file_location)
            loaded_model = torch.jit.load('uploaded_model.pt')
            output = get_model_output(model=loaded_model)
            print("Output of my Model is : ", output)
    # Read data from response.
        finally:
            if response != None:
                response.close()
                response.release_conn()
    except Exception as exp:
        print(f"Exception raised in log loop: {str(exp)}")
    sys.stdout.flush()
    sys.stderr.flush()