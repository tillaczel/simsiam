FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip3 install -r requirements.txt
WORKDIR /simsiam