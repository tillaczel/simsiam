# SimSiam: simple siamese networks for pretraining on unlabeled data
Pytorch implementation of the [SimSiam](https://arxiv.org/abs/2011.10566) paper. <br/>

[[report]("https://github.com/tillaczel/simsiam/tree/main/figures/report.pdf")] [[poster]("https://github.com/tillaczel/simsiam/tree/main/figures/poster.pdf")]

## Results
Test accuracy in percentage. Note that the [SimSiam](https://arxiv.org/abs/2011.10566) linear accuracy result is with the LARS optimizer.
<table>
  <tr>
    <th></th>
    <th>kNN acc.</th>
    <th>linear acc.</th>
  </tr>
  <tr>
    <th><a href="https://arxiv.org/abs/2011.10566">SimSiam</a></th>
    <th>∼90</th>
    <th>91.8</th>
  </tr>
  <tr>
    <th>reproduction</th>
    <th>89.42±0.19</th>
    <th>89.94±0.14</th>
  </tr>
  <tr>
    <th>80%-20% data split</th>
    <th>88.02±0.35</th>
    <th>88.34±0.21</th>
  </tr>
</table>


## Prerequisites
This repository was tested with Python 3.8.8, pytorch 1.8.0 and cuda11.1. <br/>

### Docker
For easy running I included a Docker Container. For using it please install: <br/>
[Docker](https://www.docker.com/) <br/>
[Nvidia drivers](https://www.nvidia.com/Download/index.aspx) 


## Running experiments
I assume you are using the provided Docker Image. If you are already in the environment you want to run the code in ignore the Docker commands.
### Building the Docker Image
To build the Image go to the [Docker](https://github.com/tillaczel/simsiam/tree/main/Docker) folder and execute:
```
docker build -t simsiam .
```

### Starting a training
```
docker run --rm -it --name='training' -v /home/till/PycharmProjects/representation_learning:/representation_learning --shm-size=4g --gpus all simsiam
cd simsiam
```
Insert the directory paths to `${EXP_DIR}`, `${RESULTS_DIR}` and `${DATA_DIR}`. Note if you are using your own environment you have to be at the  repository root.
```
python3 experiments/scripts/run.py experiment.exp_dir=${EXP_DIR} experiment.results_dir=${RESULTS_DIR} experiment.data_dir=${DATA_DIR}
```
For changing the hyper-parameters either use arguments:
```
python3 experiments/scripts/run.py training.max_epochs=200 experiment.exp_dir=${EXP_DIR} experiment.results_dir=${RESULTS_DIR} experiment.data_dir=${DATA_DIR}
```
or change the [config.yaml](https://github.com/tillaczel/simsiam/tree/main/experiments/scipts/config.yaml) file.

### Starting tensorboard
All the experiment results can be seen in TensorBoard. To start TensorBoard type:
```
docker run --rm -it --name='tensorboard' -p:8887:8887 -v /home/till/PycharmProjects/representation_learning:/representation_learning simsiam
tensorboard --logdir=/representation_learning/local/name/ --host 0.0.0.0 --port 8887
localhost:8887
```

### Starting visualization notebooks
Some notebooks are included to visualize the results. You need to replace the wandb paths to your models.
```
docker run --rm -it --name='notebook' -p:8886:8886 -v /home/till/PycharmProjects/representation_learning:/representation_learning simsiam
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
Go to browser and type:
```
localhost:8886
```