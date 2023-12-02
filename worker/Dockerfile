FROM python:3.9
RUN  mkdir /app
WORKDIR /app

RUN  pip3 install --upgrade redis requests minio torch
COPY worker.py /app
ENTRYPOINT ["python3", "worker.py"]