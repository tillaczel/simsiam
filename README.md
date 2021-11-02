# Representation Learning

## Prerequisites
[Docker](https://www.docker.com/) <br/>
[Nvidia drivers](https://www.nvidia.com/Download/index.aspx)

## Build Docker

GPU:
```
docker build -t representation_learning .
```


## Run training
GPU:
```
docker run --rm -it --name='training' -v /home/till/PycharmProjects/representation_learning:/representation_learning --shm-size=4g --gpus all representation_learning

```

## Start TensorBoard
All the experiment results can be seen in TensorBoard. To start TensorBoard type:
```
docker run --rm -it --name='tensorboard' -p:8888:5678 -v /home/till/PycharmProjects/representation_learning:/representation_learning representation_learning
tensorboard --logdir=/representation_learning/local/name/ --host 0.0.0.0 --port 5678
```
Go to browser and type:
```
localhost:8888
```