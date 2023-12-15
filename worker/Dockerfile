FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ENV DEBIAN_FRONTEND noninteractive
#
# RUN apt update
# RUN apt install -y vim git ffmpeg sudo libsm6

# upgrade pip
RUN python3 -m pip install --upgrade pip

# copy requirements
COPY ./requirements.txt /requirements.txt
COPY ./torchboard /torchboard

# install all required python packages
RUN pip install -r /requirements.txt

# install torchboard
WORKDIR /torchboard
RUN ./install.sh
WORKDIR /

# remove requirements
RUN rm -rf /requirements.txt

# working directory
RUN mkdir -p /workspace
COPY ./worker.py /workspace/worker.py

WORKDIR /workspace

CMD ["python3", "worker.py"]


