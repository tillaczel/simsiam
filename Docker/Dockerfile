FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip3 install -r requirements.txt
WORKDIR /simsiam