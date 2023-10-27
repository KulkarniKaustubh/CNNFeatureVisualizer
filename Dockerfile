FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt update
RUN apt install -y vim git ffmpeg sudo libsm6

# upgrade pip
RUN usr/bin/python3 -m pip install --upgrade pip

# copy requirements
COPY ./requirements.txt /requirements.txt

# install all required python packages
RUN pip install -r /requirements.txt
